import numpy as np
## build a QApplication before building other widgets
import pyqtgraph as pg
pg.mkQApp()
## make a widget for displaying 3D objects
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
view = gl.GLViewWidget()
view.show()

## create three grids, add each to the view
xgrid = gl.GLGridItem()
xgrid.setSize(x=8, y=8, z=8)
ygrid = gl.GLGridItem()
ygrid.setSize(x=8, y=8, z=8)
zgrid = gl.GLGridItem()
zgrid.setSize(x=8, y=8, z=8)
view.addItem(xgrid)
view.addItem(ygrid)
view.addItem(zgrid)

# rotate x and y grids to face the correct direction
# xgrid.rotate(90, 0, 1, 0)
# ygrid.rotate(90, 1, 0, 0)

data = gl.GLScatterPlotItem()
data.setData(pos=np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]), color=np.array([[0.5, 0.2, 0.3, 0.9], [0.2, 1, 0.3, 0.6], [0.5, 0.7, 0.2, 0.5]]), size=np.array([[4], [4], [4]]))

# scale each grid differently
# xgrid.scale(0.2, 0.1, 0.1)
# ygrid.scale(0.2, 0.1, 0.1)
# zgrid.scale(0.1, 0.2, 0.1)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



