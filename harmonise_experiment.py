# import pygame
# import pygame_menu
import numpy as np

# import socialmalicious_behaviour


import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(111)
		super(MplCanvas, self).__init__(fig)

		self.axes.set_ylim([0,20])
		self.axes.set_xlim([0,20])


class MainWindow(QtWidgets.QMainWindow):

	def __init__(self, *args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)

		self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
		self.setCentralWidget(self.canvas)

		n_data = 50
		self.xdata = list(range(n_data))
		self.ydata = [random.randint(0, 10) for i in range(n_data)]

		# We need to store a reference to the plotted line
		# somewhere, so we can apply the new data to it.
		self._plot_ref = None
		self.update_plot()

		self.show()

		# Setup a timer to trigger the redraw by calling update_plot.
		self.timer = QtCore.QTimer()
		self.timer.setInterval(10)
		self.timer.timeout.connect(self.update_plot)
		self.timer.start()



	def update_plot(self):
		# Drop off the first y element, append a new one.
		# self.ydata = self.ydata[1:] + [random.randint(0, 10)]

		# Note: we no longer need to clear the axis.
		# if self._plot_ref is None:
		#     # First time we have no plot reference, so do a normal plot.
		#     # .plot returns a list of line <reference>s, as we're
		#     # only getting one we can take the first element.
		#     plot_refs = self.canvas.axes.plot(self.xdata, self.ydata, 'r')
		#     self._plot_ref = plot_refs[0]
		# else:
		#     # We have a reference, we can use it to update the data for that line.
		# self._plot_ref.set_ydata(self.ydata)
		self.canvas.axes.cla()
		self.xdata = np.random.uniform(0,10, 100000)
		self.ydata = np.random.uniform(0,10, 100000)
		self.canvas.axes.plot(self.xdata, self.ydata, 'ro')
		# Trigger the canvas to update and redraw.
		self.canvas.draw()



app = QtWidgets.QApplication(sys.argv)



w = MainWindow()
app.exec_()


# pygame.init()
# surface = pygame.display.set_mode((1000, 800))

# def set_difficulty(value, difficulty):
#     # Do the job here !
#     pass

# def start_the_game():
#     # Do the job here !

# 	# print(menu.widget.range_slider.get_id())
# 	widget = menu.get_widget('faulty_slider')
# 	print(widget.value_changed())
# 	print(widget.get_value())

# 	widget = menu.get_widget('mal_slider')
# 	print(widget.value_changed())
# 	print(widget.get_value())


# def save_info():

# 	print('do stuff')

# 	pygame_menu.events.EXIT

# menu = pygame_menu.Menu('Welcome', 1000, 800,
#                        theme=pygame_menu.themes.THEME_DEFAULT, onclose = save_info)

# menu.add.text_input('Name :', default='John Doe')
# menu.add.selector('Difficulty :', [('Hard', 1), ('Easy', 2)], onchange=set_difficulty)
# menu.add.button('Play', start_the_game)
# menu.add.button('Quit', pygame_menu.events.EXIT)

# # Single value
# menu.add.range_slider('What percentage of robots were faulty?', default=0,
# 					  range_values = (0,10,20,30,40,50,60,70,80,90,100),
#                       rangeslider_id='faulty_slider',
#                       width = 300,
#                       value_format=lambda x: str(int(x)))

# menu.add.vertical_fill()

# menu.add.range_slider('What percentage of robots were malicious?', default=0,
# 					  range_values = (0,10,20,30,40,50,60,70,80,90,100),
#                       rangeslider_id='mal_slider',
#                       width = 300,
#                       value_format=lambda x: str(int(x)))
# menu.add.vertical_fill(min_height = 50)


# menu.add.button('Next task', start_the_game)


# menu.mainloop(surface)

# print(menu.get_widget('faulty_slider').value_changed())

# widget = menu.get_widget('faulty_slider')
# widget.value_changed()

