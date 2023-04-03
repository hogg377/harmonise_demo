
Swarm simulation demo, exploring faulty and malicious behaviours in swarms.

This demo is based off a research project by the T-B Phase group (Bristol Thales Partnership)

Contact:
elliott.hogg@bristol.ac.uk (Contact for debugging or errors in running the demo)
wenwen.gao@bristol.ac.uk
seth.bullock@bristol.ac.uk
jan.noyes@bristol.ac.uk

====================================================================================================

Dependencies are:
+   pygame for simulations (version 2.1.2)
+   pygame_menu for demo interface (version 4.2.8)
+   numpy for lots of things (version 1.19.0)
+   scipy for quickly calculating pairwise distances (version 1.5.4)
+   matplotlib for visualizing the robot simulations (version 3.3.4)

This code was written using python version 3.6

In order to install the required libraries using pip, enter the following command in the terminal.

pip3 install numpy scipy matplotlib pygame
# For linux users:
pip3 install PyQt5

If your system does not have pip3 installed, you can install it using the following command

sudo apt-get install python3-pip

*** If you have any issues running the demo, make sure you have the same versions of the dependencies as listed above. ***

	Running the demo
==========================

There are different python files that can be run for different target audiences.

openday.py is meant for the use at university open days.
schools.py is meant for outreach for schools.

In each case the wording and presentation of the demo has been tailored for each audience so that the 
purpose of the demo is clear.

Exectute "run_demo.py" to run the experiment. This will:
+ Launch pygame and the experiment interface.
+ Follow the instruction screens
+ The user will see a first trial where they only watch the simulation.
+ Then a second simulation allows the user to use the arrow keys to direct the swarm in different directions. 

Inbetween each simulation a replay is shown with the faulty/malicious robots highlighted with red stars. 
The seed is fixed for these simulations so the replays will be exactly the same as what the participant saw
previously.

The idea of the demo is that a second screen will show the "healthy" behaviour of the swarm on a loop using a video. 
A video of the healthy behaviour can be found in the "videos folder"


	Known Issues
======================

Sometimes when launching the python script, the pygame interface will appear unresponsive or frozen when
pressing any of the menu buttons or scrolling.

The window will not actually be frozen but the window needs to be dragged or moved slightly in order
for the screen to refresh. This way you can click the buttons, then move the window slightly and it 
will then show then next screen. Alternatively, closing the window and running the script again is the
only other option.

There is no current fix for this and it is unclear what is causing the issue. 














