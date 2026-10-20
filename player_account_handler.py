import csv
import os
import time
import requests
from tqdm import tqdm

LICHESS_USERS_API_URL = "https://lichess.org/api/users"


class PlayerAccountHandler:
    """Gets the account status of players from the lichess API:

    'closed' = possibly cheating
    'tosViolation' = cheating or rating manipulation
    'open' = account in good standing
    'not found' = account does not exist (this should not happen)

    From the lichess api documentation:
    All requests are rate limited using various strategies,
    to ensure the API remains responsive for everyone.
    Only make one request at a time. If you receive an HTTP response with a 429 status,
    please wait a full minute before resuming API usage.

    The users endpoint is also limited to 8,000 users every 10 minutes, and 120,000 every day.
    To stay within these limits, players are requested in batches of up to 300 (the maximum
    for the users endpoint), one request every 24 seconds (7,500 users every 10 minutes),
    and at most 110,000 users are requested per run. After a 429 response, we wait 70 seconds,
    doubling up to 10 minutes if we are still rate limited. If a cache file path is given,
    account statuses are saved to it after each request, so players are never requested twice
    and an interrupted run can be resumed.
    """

    BATCH_SIZE = 300
    SECONDS_BETWEEN_REQUESTS = 24
    SECONDS_AFTER_RATE_LIMIT = 70
    MAX_SECONDS_AFTER_RATE_LIMIT = 600
    MAX_RETRIES = 6
    MAX_USERS_PER_RUN = 110_000

    def __init__(self, cache_file_path=None):
        self._account_statuses = {}
        self._cache_file_path = cache_file_path
        if cache_file_path and os.path.exists(cache_file_path):
            with open(cache_file_path, newline="") as f:
                for player, status in csv.reader(f):
                    self._account_statuses[player] = status

    def update_player_account_status(self, player):
        self.update_player_account_statuses([player])

    def update_player_account_statuses(self, players):
        """Requests the account status of each player not already known, in batches."""
        unknown_players = list(
            dict.fromkeys(p for p in players if p not in self._account_statuses)
        )
        if not unknown_players:
            return
        if len(unknown_players) > self.MAX_USERS_PER_RUN:
            print(
                f"Warning: {len(unknown_players):,} players exceeds the daily limit of the lichess API, "
                f"so only the first {self.MAX_USERS_PER_RUN:,} will be requested. Run again tomorrow to continue."
            )
            unknown_players = unknown_players[: self.MAX_USERS_PER_RUN]

        batches = [
            unknown_players[i : i + self.BATCH_SIZE]
            for i in range(0, len(unknown_players), self.BATCH_SIZE)
        ]
        for i, batch in enumerate(tqdm(batches, desc="Requesting account statuses")):
            if i > 0:
                time.sleep(self.SECONDS_BETWEEN_REQUESTS)
            batch_statuses = self._request_account_statuses(batch)
            self._account_statuses.update(batch_statuses)
            self._save_to_cache(batch_statuses)

    def _request_account_statuses(self, players):
        """Returns {player: status} for a batch of up to BATCH_SIZE players."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    LICHESS_USERS_API_URL,
                    data=",".join(players),
                    headers={"Content-Type": "text/plain"},
                    timeout=60,
                )
            except requests.exceptions.RequestException as e:
                print(f"Request failed ({e}), retrying...")
            else:
                if response.status_code == 200:
                    break
                print(f"Received HTTP {response.status_code} from lichess, retrying...")
            if attempt == self.MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Could not get account statuses after {self.MAX_RETRIES} attempts"
                )
            time.sleep(
                min(
                    self.SECONDS_AFTER_RATE_LIMIT * 2**attempt,
                    self.MAX_SECONDS_AFTER_RATE_LIMIT,
                )
            )

        ## lichess user ids are lowercase usernames, and closed accounts are returned as disabled
        users_by_id = {user["id"]: user for user in response.json()}
        account_statuses = {}
        for player in players:
            user = users_by_id.get(player.lower())
            if user is None:
                account_statuses[player] = "not found"
            elif user.get("tosViolation"):
                account_statuses[player] = "tosViolation"
            elif user.get("disabled"):
                account_statuses[player] = "closed"
            else:
                account_statuses[player] = "open"
        return account_statuses

    def _save_to_cache(self, account_statuses):
        if self._cache_file_path:
            with open(self._cache_file_path, "a", newline="") as f:
                csv.writer(f).writerows(account_statuses.items())
