import json
import os
from plotly.offline import get_plotlyjs_version

PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-{plotlyjs_version}.min.js"></script>
<style>
  body {{ font-family: "Open Sans", verdana, arial, sans-serif; color: #2a3f5f; background: white; margin: 16px; }}
  .dropdowns {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 8px 0; }}
  label {{ font-size: 13px; }}
  select {{ margin-left: 6px; font-size: 14px; padding: 4px 6px; border: 1px solid #c8d4e3; border-radius: 4px; background: white; }}
</style>
</head>
<body>
<div id="header-figure"></div>
<div class="dropdowns" id="dropdowns"></div>
<div id="figure"></div>
<script>
const headerFigure = {header_figure_json};
const entries = {entries_json};
const dropdownLabels = {dropdown_labels_json};
const traceFilter = {trace_filter_json};

const selects = [];
function addDropdown(label) {{
  const wrapper = document.createElement("label");
  wrapper.textContent = label;
  const select = document.createElement("select");
  select.addEventListener("change", update);
  wrapper.appendChild(select);
  document.getElementById("dropdowns").appendChild(wrapper);
  return select;
}}
dropdownLabels.forEach((label) => selects.push(addDropdown(label)));
let filterSelect = null;
if (traceFilter) {{
  filterSelect = addDropdown(traceFilter.label);
  Object.keys(traceFilter.options).forEach((option) => filterSelect.add(new Option(option, option)));
}}

function update() {{
  // each dropdown only lists the options that exist for the options chosen in the dropdowns before it
  for (let i = 0; i < selects.length; i++) {{
    const matching = entries.filter(([key]) => selects.slice(0, i).every((select, j) => key[j] === select.value));
    const options = [...new Set(matching.map(([key]) => key[i]))];
    const previous = selects[i].value;
    selects[i].replaceChildren(...options.map((option) => new Option(option, option)));
    selects[i].value = options.includes(previous) ? previous : options[0];
  }}
  const [, figure] = entries.find(([key]) => selects.every((select, i) => key[i] === select.value));
  let data = figure.data;
  if (traceFilter) {{
    const shown = traceFilter.options[filterSelect.value];
    data = data.map((trace) => ({{ ...trace, visible: shown.includes(trace.meta) }}));
  }}
  Plotly.react("figure", data, figure.layout, {{ responsive: true }});
}}

if (headerFigure) {{
  Plotly.newPlot("header-figure", headerFigure.data, headerFigure.layout, {{ responsive: true }});
}}
update();
</script>
</body>
</html>
"""


def to_json(value):
    """Returns value as json that is safe to put inside a script tag."""
    return json.dumps(value).replace("</", "<\\/")


def write_dropdown_page(
    file_path,
    page_title,
    figures,
    dropdown_labels,
    trace_filter=None,
    header_figure=None,
):
    """Writes one html page that shows one of several figures at a time, chosen with a dropdown
    for each part of the figure keys.

    figures maps a tuple of option labels, one per dropdown label, to a plotly figure, and the
    options are listed in the order of the figures. trace_filter = {"label": ..., "options":
    {option label: [trace meta values to show]}} adds a dropdown that shows only the traces of
    the figure whose meta is in the chosen option. header_figure is always shown above the dropdowns.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    entries = [
        [list(key), json.loads(figure.to_json())] for key, figure in figures.items()
    ]
    with open(file_path, "w") as f:
        f.write(
            PAGE_TEMPLATE.format(
                page_title=page_title,
                plotlyjs_version=get_plotlyjs_version(),
                header_figure_json=to_json(
                    json.loads(header_figure.to_json()) if header_figure else None
                ),
                entries_json=to_json(entries),
                dropdown_labels_json=to_json(dropdown_labels),
                trace_filter_json=to_json(trace_filter),
            )
        )
