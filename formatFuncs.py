import numpy as np

def fmt_yr(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    a = int(a)
    if a!=1:
        return r'${} \times 10^{{{}}}$ yr'.format(a, b)
    else:
        return r'$10^{{{}}}$ yr'.format(b)
    
def fmt_tres(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    a = int(a)
    if a!=1:
        return r'$t_{\mathrm{res}}=$' + r'${} \times 10^{{{}}}$ yr'.format(a, b)
    else:
        return r'$t_{\mathrm{res}}=$' + r'$10^{{{}}}$ yr'.format(b)
    
    
def fmt_tbin(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    a = int(a)
    if a!=1:
        return r'$t_{\mathrm{bin}}=$' + r'${} \times 10^{{{}}}$ yr'.format(a, b)
    else:
        return r'$t_{\mathrm{bin}}=$' + r'$10^{{{}}}$ yr'.format(b)
    
    
def add_arrow(line, position=None, direction='right', size=15, color=None, label='',xtext=1.0,ytext=1.2):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = -1 #start_ind + 1
        end_rat = 2
    else:
        end_ind = 0 #start_ind - 1
        end_rat = 0.5

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[start_ind]*end_rat, ydata[start_ind]),
        arrowprops=dict(arrowstyle="->",color=color,linewidth=3),
        size=size
    )
    if direction == 'right':
        line.axes.annotate(label,
            xytext=(xdata[start_ind]*xtext, ydata[start_ind]*ytext),
            xy=(xdata[start_ind]*xtext, ydata[start_ind]*ytext),
            color=color,
            fontsize=size
        )
    else:
        line.axes.annotate(label,
            xytext=(xdata[start_ind]*end_rat*xtext, ydata[start_ind]*ytext),
            xy=(xdata[start_ind]*end_rat*xtext, ydata[start_ind]*ytext),
            color=color,
            fontsize=size
        )