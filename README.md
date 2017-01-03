# 3d-printer-weaver
Empower your 3d-printer with string art.


Happy New Year 2017 :)

This project won't be developped further anytime soon, but If you'd like to support us, you can do so at our website. https://gistnoesis.net

#TLDR: How to use the soft at the bottom. The thiner the thread the better. Multiple small improvements have a large impact.



#The story :

I saw https://github.com/theveloped/ThreadTone last week on HN, and decided I could mod my 3d printer as a 3 day week-end project, to do the same thing.
I have a Taz-4 but it should work with a lot of open-source printers.
Because the workarea is smaller with a 3d printer than with a laser cutter, it presented quite a few interesting challenges to miniaturize its solution.
On the other hand, as I'm printing the circular loom, it's already precisely positionned, and I can have tighter clearances.

I started the first day and achieved pretty promising results quite quickly.
I used openscad to design the circular loom. 
I began with 50 2mm height, 1mm radius, pillars in a 70mm radius ring, and wrote a simple algorithm to generate the trajectory of the head.
To feed the thread I used an old pen ink plastic tube (3mm wide), which I taped to the printer head.


![Prime Rings](/pictures/hyperbolicRings.jpg?raw=true "Prime Rings")

It has 100 lines it is a symetric desgn which was constructed algorithmically, by choosing the distance between consecutive pillars (13, and 17) so that they are coprime with the number of pillars, so we touch all pillars exactly one time.


On the second day, I worked on being able to weave an image. 

Quickly I realized that the original ThreadTone project, needed at least 150, (300 better) anchor points to have correct results.
Also the algorithm used in ThreadTone is far from being optimal, as some ThreadTone comments revealed there is an alternative project 

https://github.com/danielvarga/string-art which uses a mathematical technique (solving a linear equation) to obtain the optimal result when the wires are infinitely thin.

string-art needs its input in greyscale negative (on ubuntu we can use : convert -negate -colorspace gray imginput.jpg imgoutput.jpg) to obtain the correct input.

input :
![Olympic input](/pictures/olympic-neg.jpg?raw=true "input to string-art")


output :
![Result](/pictures/olympic-unquantized.png?raw=true "output of string-art")

We can draw more detailed images too :


![Vitruve](/pictures/vitruve2-unquantized.png?raw=true "Vitruve")



I decided to use string art. It produces a graph of around 3000 edges as a result, and not all lines have the same darkness so it means that some lines must be repeated to increase the contrast.
String art use a quantisation formula to convert the floating point results of the linear solver, into a discrete number of thread. 
But when there is high contrast it happens that some times the edge must be repeated 15 times, which poses problem during manufacturing.

After trying a few times to modify the quantisation algorithm. I choose to reduce the number of edges by using a sampling strategy, montecarlo fashion. 
This technique also permits me to choose the number of edges, and allows for probabilistically accurate reducing of the number of edges. It also means we could continue add edges to a current weaving by overlaying a new graph on top of a previous graph.
I increased the number of pillars to 150 with a 70mm radius ring. And tried to weave. 


![150 pointy pillars](/pictures/150pointypillars.jpg?raw=true "150 pointy pillars in a 70 mm radius ring")


Because of the number of pillars and the small size of the ring, it means that there is less than 1mm between the pillars. So the thread guide can't cross. So it must jump over the pillars, go down, hook the pillar, then up again, jump the pillar, then down again.
I tweaked my trajectory algorithm to take this into account. It kind of work and succeded to hook the pillars 3 out of 4 times, which means than when you have thousand of pillars to hook that it won't work.
The flexible ink-pen tube, and tape, was not precise enough.

So I looked for an alternative guide. Luckily I found a needle to inflate soccer balls. It was 2mm wide, and had a hole all the way though so I can pass a needle through it to insert the thread into it.
To attach the needle to the printing head I tried tape, but it proved ineffective. So I designed a needle holder in openscad :
A 10mm cube with a hollow cylinder in the middle. And printed it, to attach it to the head, I remarked that when the printing head is still hot ABS will stick to it. 

![Needle Mount](/pictures/needleMount.jpg?raw=true "Mounting the needle to the hot head")

The needle fit is quite tight (automatically adjusted by the hot head trick). The needle must be below the printing head so the printing head don't hurt the pillars, and not to far from the printing head so that it don't reduce the print area much.
Also when homing the printer, the needle fit should not be so strong that it would break the glass plate. But it turns out it worked just right.

After these adjustments, It worked better, but not quite well enough. In particular as more and more threads go in, it becomes harder to add more without tripping other wires. 

At the end of the second day, results were unsatisfactory, but the needle holder was a very definite improvement. It also reveals some problems: when I add to much thread it becomes too dark, and we can't see through anymore. 

On the next day, I decided to mitigate those two problems with one stone. Increasing the ring size. It now is 110mm radius, with the 150 pillars the needle can go between without the need to jump, with a 0.5mm clearance. 
It worked better, but there were some visual artifacts (mainly blurryness but which is important because the effect we are trying to observe are subtle), because the pillars are of the same order of size as the distance between them, the approximation of them as a single point don't hold. So instead for each pillar I now have two anchor points,
either the top side or the bottom side of the pillar. Which means that with 150 pillars, I can have the equivalent of a "300 anchor points" resolution.

I had to tweek string-art to add these unevenly spaced anchor points.


I adapted the algorithm to the increase resolution and weaving constraints. 
The weaving was fine but it became too dark as lines where added. 
So I used a smaller diameter polyesther thread, normally used for sewing machines. It immediately worked a lot better.

I tried to weave 800 lines which was a length of 196 meters, I have a 200 meters roll, I had to stop at 61% so around 150 lines, because the needle was starting to trip the wires (it can probably be mitigated by augmenting the z of the needle thoughout the weave and increasing the height of the pillars). Also some pillar broke due to repetitive pulling (more precisely were deformed enough that the needle touched them on the next travel) ,  
The results could also be improved by using even more thin threads, some wider, and taller pillars, but a little less of them.

Here is what I obtain :

![Olympic Rings](/pictures/olympicRings.jpg?raw=true "Olympic Rings")




Lots of possible improvements can be made.
To go further :
http://www.sosafresh.com/3d-weaver/
https://www.youtube.com/watch?v=iFLZEnoDHe8




#Here's how to use the soft :

dependencies :
pip install numpy matplotlib pylab

use my version of string-art it has 150 pillar and 1mm pillar radius hardcoded (inside the "atan" ), so 300 anchor points, you can also adjust the hardcoded shrinkage to increase or reduce the size of the image :

python string-art.py input.jpg output-prefix          #it takes 5 min using a single core, although it is much faster with 2*less pillars (probably an issue with the original string-art using dense matrix instead of sparse matrix in the solver)


then use ipython 

import wiring
l,g = wiring.loadGraphFromFile( "output-prefix-unquantized.txt", brokenPins , 40, nbEdges ) # use brokenPins = {} if no pins are broken, or set the broken pillars hardcoded in wiring.brokenPin() and pass this dictionary

you can then view the graph with 

wiring.displayGraph2( 150, g, 110, linewidth = 0.1)  # 150 = number of pillars

traj = wiring.generateGraphTrajectory2( g , 150, radius_of_the_ring, 3.0 ) #3 = 1.2mm pillar radius + 1mm radius head tool + 0.8mm clearance

wiring.writeGcode( traj, outname ) 

I have a LulzBot Taz4 so I emmitted some marlin gcode, but it's pretty standard and should work anywhere, I don't set speed in it so set your speed in your printer interface)
#The Gcode positioning is relative so we can precisely postion the needle to take into account the offset between hothead and needle


To 3d-print you can use openscad to generate the STL from the scad file.
Then you slice it with slicer.
Then you 3d print and leave the part in place.
While the hot head cools you stick the needle holder to it

At the end of the 3d print you manually move the needle (with the thread through it) so that the bottom of the needle is exactly between the two rightmost pillars
 (theta=0, r= radiusof the ring,z = 1.5mm above the ring), it should be below the top of the pillar able to move 1mm  down without touching the ring.

You attach the thread to the pillar 0, one pillar in the counter clockwise from the needle. 
You load the gcode, and print.  For your first tries you can increase z so it moves over the pillar without touching them to check that everything is OK.
 (If it is going slow, set the "modal" speed by doing a fast back and forth ( +10 -10 ) of the head to make it switch to fast mode. )

Additional tip : if the wire is not broken and the weaving failed you can rewind the thread quickly (lift the needle 10 cm above the center of the ring and rewind), so you can try again. 
I left a lot of dead code in the code source so you can follow my train of thought. 
