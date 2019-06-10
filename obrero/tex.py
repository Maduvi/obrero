import pandas as pd

from obrero import enso


def oni_to_tabular(dfoni, texname):
    """Turn ONI data frame into TeX table.

    Given an ONI pandas dataframe, creates a .tex file with tabular 
    environment that uses CTAN 'booktabs' and 'xcolor' packages.

    Parameters
    ----------
    dfoni: pandas.DataFrame
        This data frame contains ONI values. This should have been
        created with function `enso.get_oni`.
    texname: str
        Name of the output .tex file.
    """ # noqa

    # make sure it is a pandas dataframe
    if not isinstance(dfoni, pd.core.frame.DataFrame):
        raise ValueError('input must be a pandas dataframe')

    # get dimensions
    nyear, nmon = dfoni.shape

    # get enso phases (to define colors)
    dfenso = enso.enso_finder(dfoni)

    # get header (with extra column at beginnning)
    cols = dfoni.columns
    head = [r'\textbf{%s}' % x for x in cols]
    header = r'&' + r'&'.join(head) + r'\\' + '\n'

    # open file instance
    fout = open(texname, 'w')

    # start writing TeX code
    fout.write(r'\begin{tabular}{ccccccccccccc}' + '\n')
    fout.write(r'\toprule' + '\n')
    fout.write(header)
    fout.write(r'\midrule' + '\n')

    # now create every line separately (to color enso)
    for i in range(nyear):

        # empty list for line to fill every time
        parts = []

        # get year and append it
        year = dfoni.iloc[i].name
        parts.append(r'\textbf{%4i}' % year)

        # now check every number to color it
        for j in range(nmon):

            oni = dfoni.values[i, j]
            phase = dfenso.values[i, j]

            # oni formatting
            if phase == 1:
                parts.append(r'\textcolor{red}{%3.1f}' % oni)
            elif phase == -1:
                parts.append(r'\textcolor{blue}{%3.1f}' % oni)
            else:
                parts.append('%3.1f' % oni)

        # construct single line and write it
        line = '&'.join(parts) + r'\\' + '\n'
        fout.write(line)

    # close environment
    fout.write(r'\bottomrule' + '\n')
    fout.write(r'\end{tabular}' + '\n')

    print('Created file: \'' + texname + '\'')
    fout.close()
