import re

def get_label(titleline):
    """
    Extract label from title line in begin environment.
    Return label and title (without label).
    """
    label = ''
    if 'label=' in titleline:
        pattern = r'label=([^\s]+)'
        m = re.search(pattern, titleline)
        if m:
            label = m.group(1)
            titleline = re.sub(pattern, '', titleline).strip()
    return label, titleline

def latex_minipage(text, titleline, counter, format):
    """LaTeX typesetting of admon-based "minipage" environment."""
    label, titleline = get_label(titleline)
    s = r"""
\begin{minipage}[\columnwidth]
"""
    if label:
        s += 'label{%s}\n' % label  # no \ (is added by DocOnce)
    s += r"""
%s
\end{minipage}
""" % text
    return s

def html_minipage(text, titleline, counter, format):
    """HTML typesetting of admon-based "minipage" environment."""
    label, titleline = get_label(titleline)
    s = r"""
<quote style="font-size: 80%%">
"""
    if label:
        s += '<a name="%s"></a>\n' % label
    s += r"""
%s
</quote>
""" % (text)
    return s


def do_minipage(text, titleline, counter, format):
    """General typesetting of minipage environment via a section."""
    label, titleline = get_label(titleline)
    s = """

===== %s =====
""" % (titleline)
    if label:
        s += 'label{%s}\n' % label
    s += '\n%s\n\n' % text
    return s


envir2format = {
    'intro': {
        'latex': r"""
""",},
    'minipage': {
        'latex': latex_minipage,
        'html': html_minipage,
        'do': do_minipage,
        },
}
