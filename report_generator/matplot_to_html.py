import io
from matplotlib.figure import Figure

def fig_to_html(fig: Figure) -> str:
    imgdata = io.StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    ##strip the xml doctype then embedd it into the html
    svg_string = imgdata.getvalue()
    header = """<?xml version="1.0" encoding="utf-8" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">"""
    assert svg_string[:len(header)] == header
    return svg_string[len(header):]
