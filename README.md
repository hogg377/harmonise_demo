python implementation of the matlab model used for simulating grid world herding
Dependencies are:
+   pygame for visualisation
+   numpy for lots of things
+   scipy for quickly calculating pairwise distances
+   opencv for recording videos of simulation runs
+   treelib for creating branching trees of potential paths used in the empowerment calculation
+   matplotlib (only needed for empowermentTest) to visualise empowerment values for a static set of positions

```bash
pip3 install numpy scipy matplotlib pygame treelib opencv-python==4.3.0.36
# For linux users:
pip3 install PyQt5
```

Exectute "Visualise.py" to run the model.  This will:
+   Create a "World" which records the position of all the agents
+   Create a pygame screen to visalise where the agents are
+   Create a pack "Population" and a flock "Population"
+   Create two "Sheep" and one "Dog", both a subclass of "Agent"
+   Setup everything using the config in "config.py"
+   Run the model in a pygame style loop until the user closes the pygame screen window

The stucture of the main loop is:
1.  Call "update" on the pack and flock populations.
2.  The populations then call "update" on their respective agent types
3.  The agents update their view of the world and decide where they want to move
4.  An agent moves by seting its new position in the World's self.m_next_grid.
    The World may not allow the move if it's outside the boundary or occupied by another agent.
    The agent may then ask what positions it could move to and pick one of those instead.
5.  Once an agent has finished its update, the population will also update the agent's sprite
    to match the agent's position.
6.  One all agent's have moved, the World advances one tick by copying it's self.m_next_grid
    to self.m_grid and self.m_grid to self.m_grid_previous
7.  The screen is redrawn and updated on screen.
8.  back to 1.

Current status:
---------------
Uses the genome and starting positions from matlab run: 20210803T212146
All flock and pack agents are homogeneous
+   30/08/21: GridEmpowerment added to calculate an N step look ahead of empowerment.  Empowerment values are visualised as numbers next to each dog agent.  In empowermentTest.py, a set of dog and sheep positions can be set and the empowerment viewed for each square moved from P0.  A black square is high empowerment (255), white is low empowerment (0), and greyscale for intermediate values.
+   23/09/21: Added functions to spawn and delete dogs, pause/unpause and change color of dog in response to their empowerment value
+   27/09/21: Added function: pressing keys 1,2,3 or 4 will change the method the dogs use to calculate their empowerment.  The change is immediately visible on the screen and works while the simulation is paused.
+   02/11/21: Merged code from MC and CB to create a heuristic based dog. 
+   05/11/21: heuristic_dog branch appears working but needs testing.  A minimum of 4 dogs are needed at all times
+   12/11/21: added max and min dog limits to the config and function to Visualise to apply them.
+   12/11/21: added if statement to calculateFlockFeatures to prevent crash when there are less than 3 dogs.  If there are 2 dogs, the sheep are partitioned based on which is closest to each sheep (very crude and slow method).  If there's only one dog then all sheep are in the dogs partition.
+   12/11/21: added score calculated as the time integral of 1 / (distance from flock CoM to goal * number of dogs present)
+   12/11/21: added a fix, if the dog is on the waypont then the error signal is zero (this is needed mainly to account for the quantisation error between the grid square vs true waypoint).  Fix should stop dogs from dithering between squares when next to a sheep which is moving slower than they are.
+   12/11/21: added field to cfg['dog'] called dog_2_sheep_speed which sets the ratio of ticks a sheep moves on relative to a dog.  e.g., a value of 1 means both sheep and dog move on alternatve ticks (i.e. move at same speed), a value of 2 means the dogs move on ticks i, i+1 and then the sheep moves on tick i+2.
+   12/11/21: added "spawn_radius" field to cfg['dog'] to set the region over which a dog will appear following a mouse click.  A value of zero means the dog appears exactly where the user clicks, a value of x means it's click_position + [ [-x,+x], [-x,+x] ]  where [-x,+x] denotes a randomly selected integer in the range -x to +x
+   13/11/21: changed the targeting behaviour when a dog is driving.  Previous approach attempted to drive the flock CoM but this would only work if sheep interactions caused the flock to form a cohesive group (it doesn't!).  New approach targets the furthest sheep (in a dogs partition) from the goal and drives it towards the goal during driving.  This compares with driving the furthest (partition) sheep towards the CoM when collecting.  Also changed the condition for switching between collecting and driving to use a threshold based on the # of the sheep
+   13/11/21: the dog will not select a sheep as a target if that sheep is within a minimum threshold distance of the goal
+   13/11/21: added field to cfg['dog] called "show_empowerment_values_b".  If set true, it will query the dogs and display their current empowerment as a numerical value
+   13/11/21: added mean_empowerement as the average empowerment per dog per tick.  This and score are for each tick and recorded as vectors.  Added plot to display at end of the simulation to show their progression througout the game.
+   20/11/21: changed sumEmpowerment() in GridEmpowerment.py to call a new function isImprovedState.  If task_weight_empowerment_b is passed as true then only states which reduce the cumulative total distance of the sheep from the goal will be counted as adding to empowerment.  Works for any of the current methods for summing empowerment e.g. leaves only, unique states only etc...
+   20/11/21: added variable m_empowerement_task_weighted_b to class Dog.  It defaults to false but can be toggled by pressing 0 (also works when the game is paused)
+   29/11/21: added sheep parameter use_flocking_behaviour to config.  When set to true Sheep the sheep will have an addtional contribution to its movement.  This is calculated by the new function "respondToHerding()".  The net effect is that sheep will tend to move towards their local CoM and "stick" to other sheep making them easier to move as a group.


Bugs:
---------------
+   30/08/21: Empowerment values are surprisingly high when the sheep is trapped against the world boundary.  It may be because the world model in GridEmpowerment is very dumb.  It may be better to embed a world model in the dog (it already has one in the matlab version), and then call GridEmpowerment from inside the dog (so we can use the dog's model of the world).  Dog would need some changes to enable this
+   30/08/21: if the simulation is visualised at high speed, it looks like the dog maintains a single square from a sheep.  It think this is just a visual quirk from the way the screen redraws and the longer time it takes for a dog to update than a sheep. i.e. a dog tick takes longer than a sheep tick, but needs checking.
+   03/11/21: Dog is able to reach edge of world and causes a crash when it does (usual reference from 1 instead of 0 problem) - FIXED
+   05/11/21 in World, isEmptyGridPosition() will pass a world position to isValidGridPosition() to prevent a reference from 1 being converted to a reference from 0 twice, this is confusing and needs a better fix long term


Bugs fixed:
---------------
+   23/09/21: Sometimes crashes when sheep get close to the boundaries because it references a location outside the grid. - FIXED World.isAgentMoved() now converts position to a grid reference from 0 before accessing the world grid.
+   10/09/21:isValidSquare() in grid empowerment was classifying a square on the edge of the local state as out of bounds.
+   10/09/21:calcMovementTree() the sight_horizon was increased from max_moves+1 to max_moves+2 to enable a sheep on the local state boundary to move one step away on the final step of the dog.
+   10/09/21: in empowermentTest, when a move to a child node causes a change in the SoTW, the empowerment value is added to the Parent node NOT the child.
+   10/09/21: in World, isValidGridPosition should have checked position was an integer before converting from a position (referenced from 1) to a grid coordinate (referenced from 0)
+   10/09/21: in World, isValidGridPosition should have checked the grid coordinate was <=(world width-1) not <(world width-1)
+   10/09/21: in World, entitiesInRange the search was stopping 1 short of the upper limit due to python's slicing range returning 1 less than the upper limit
+   21/09/21: In World, entitiesInRange, the search grid was 1 square too large in both dimesions due to the previous bug fix adding +1 in 2 locations and not one.  (can't get the staff)
+   03/11/21: In Dog, added catches to selectTargetSheep and selectSteeringPoint to handle the case where no sheep appear in the dog's partition
+   05/11/21: Video recorder now rotates the frame so playback is the right way round
+   05/11/21: Changed go_home condition to account for the flock com distance from the goal AND the distance the furthest sheep is from the com
+   05/11/21: the heuristicDog function "planState" now uses all visible sheep to decide its state (not just those in its partition)
+   05/11/21: cosineStep function in heuristicDog replaced with a grid based version taken from CB's latest NN version.  Returns a value in the range [-1,+1] using the dot product and the sign of the cross product.  Output sets the amount the dog will circulate a sheep
+   05/11/21: removed the sheep repulsion term from the dog's herding behaviour (it won't avoid the wrong sheep anymore but instead plough through the flock, this might be ok given the interaction between dogs and sheep is very close range)
+   05/11/21: added checks in heuristicDogs herdingBehaviour to prevent the dot product angle returning nan (divide by 0 or acos(|x|>1))
+   05/11/21: heuristicDog now remembers it's herding state.  It also remembers the distance to the CoM when it first encountered its current target
+   05/11/21: in World, isEmptyGridPosition() will pass a world position to isValidGridPosition(), this is confusing and needs a better fix long term
+   05/11/21: many other minor bug fixes!!
+   08/11/21: fixed but where angles in the range 337-360 was quantised to NE instead of N
+   12/11/21: fixed in vector2CompassDirection(vec) where vec=[0,0] was returning a direction of [+1,0] instead of [0,0]
+   13/11/21: if the voronoi partition fails, partitionSheep() will return an empty partition instead of raising an except and causing the program to terminate




