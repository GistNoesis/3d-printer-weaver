
import math
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import pylab as pl

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def cart2pillar(x, y,nbPillar):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi / (2*math.pi) * nbPillar)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def circleAroundPillar(rc,theta, rp,dir):
    out = []
    up = 1.0
    down = 0.0
    initangle = math.pi + theta
    nbsteps = 20
    dt = 2.0*math.pi / (nbsteps)
    for i in range(nbsteps+1):
        ang2 = initangle+ dir*dt*i
        pc = (rc * math.cos( theta ),rc*math.sin(theta))
        if (float)(i) / nbsteps > 0.9 or (float)(i) /nbsteps < 0.1:
            z = up
        else:
            z = down
        out.append( np.array([ pc[0]+ rp*math.cos(ang2) , pc[1] + rp* math.sin(ang2),z]) )

    return out

def getPillarAngle( pillar, nbPillar ):
    return 2.0*math.pi / (nbPillar) * ( pillar + 0.5)

def generateTrajectoryBetweenPillar(pillar,nbPillar,rc,rp,dir):
    #we go to the inner circle middle angle
    angle = (getPillarAngle(pillar,nbPillar) + getPillarAngle(pillar-1,nbPillar) ) / 2.0
    out = []
    pin = ( (rc-dir*rp)*math.cos(angle),(rc-dir*rp)*math.sin(angle) )
    pullout = ( (rc+dir*1.5*rp)*math.cos(angle),(rc+dir*1.5*rp)*math.sin(angle) )
    pout = ( (rc+dir*rp)*math.cos(angle),(rc+dir*rp)*math.sin(angle) )
    up = 3.0
    mid = 0.0
    down = -1.0

    if dir > 0.0 :
        out.append( np.array( [pin[0],pin[1],up] ) )
        out.append(np.array([pin[0], pin[1], mid]))
        #out.append(np.array([pin[0], pin[1], up]))
        #out.append(np.array([pout[0], pout[1], up]))
        out.append(np.array([pout[0], pout[1], down]))
    else:
        out.append(np.array([pin[0], pin[1], down]))
        #out.append(np.array([pin[0], pin[1], up]))
        out.append(np.array([pout[0], pout[1], mid]))
        out.append(np.array([pout[0], pout[1], down]))
        out.append(np.array([pullout[0], pullout[1], down]))
        out.append(np.array([pullout[0], pullout[1], up]))
        #out.append(np.array([pout[0], pout[1], up]))
    return out

def generateOutMoveTrajectory(pillar1, pillar2, nbPillar,rc,rp,dir):
    print "out move from " + str(pillar1) + " to " + str(pillar2)
    angle = (getPillarAngle(pillar1, nbPillar) + getPillarAngle(pillar1 - 1, nbPillar)) / 2.0
    targetangle = (getPillarAngle(pillar2, nbPillar) + getPillarAngle(pillar2 - 1, nbPillar)) / 2.0

    while 0 < dir * ( angle - targetangle ):
        targetangle = targetangle + dir* 2.0*math.pi

    dangle = dir*2.5 * math.pi / 180.0
    out = []

    while dir * (angle - targetangle) < 0:
        p = ((rc + rp) * math.cos(angle), (rc + rp) * math.sin(angle))
        out.append( np.array( [ p[0],p[1], -1.0 ] ) )
        angle += dangle

    p = ((rc + rp) * math.cos(targetangle), (rc + rp) * math.sin(targetangle))
    out.append(np.array([p[0], p[1], -1.0]))
    return out

def hookPillar( pillar, nbPillar, rc,rp):
    out = []
    out.extend( generateTrajectoryBetweenPillar( pillar,nbPillar,rc,rp, 1.0) )
    out.extend( generateOutMoveTrajectory( pillar, pillar-1, nbPillar,rc,rp,-1.0) )
    out.extend( generateTrajectoryBetweenPillar( pillar-1,nbPillar,rc,rp,-1.0) )
    return out

def jumpPillar( pillar1, pillar2, nbPillar,rc,rp,dir):
    out = []
    out.extend( generateTrajectoryBetweenPillar( pillar1,nbPillar,rc,rp, 1.0) )
    out.extend( generateOutMoveTrajectory(pillar1, pillar2 , nbPillar, rc, rp, dir))
    out.extend( generateTrajectoryBetweenPillar(pillar2, nbPillar, rc, rp, -1.0))
    return out

#common rp is 4mm
def generateTrajectory( pillarseq, nbPillar,rc, rp):
    out = [ np.array( [rc,0.0,0.0] ), np.array( [rc-rp,0.0,0.0] ) ]
    for ind in pillarseq:
        out.extend( circleAroundPillar(rc,getPillarAngle( ind, nbPillar),rp,1.0 ))
    out.append( np.array( [rc-rp,0.0,1.0] ) )
    out.append(np.array( [rc, 0.0, 0.0]) )
    return out

def writeGcode(traj,outname):
    with open(outname,"w") as f:
        f.write("G91\n")
        f.write("G21 ; set units to millimeter\n")
        #f.write("M204 S900;\n")
        f.write("G00;\n")
        for i in range(1,len(traj) ):
            diff = traj[i]-traj[i-1]
            print diff
            if math.fabs(diff[2])> 1e-6 :
                f.write(str.format( "G00 X{0:.3f} Y{1:.3f} Z{2:.3f}\n", diff[0],diff[1],diff[2]))
            else:
                f.write(str.format("G00 X{0:.3f} Y{1:.3f}\n", diff[0], diff[1]))


def removeEdgeIfWeighIs0( g, n1,n2):
    if (g[n1][n2] == 0):
        if (len(g[n1]) == 1):
            g.pop(n1)
        else:
            g[n1].pop(n2)

def decreaseWeightEdge( g, n1,n2):
    g[n1][n2] = g[n1][n2] - 1
    g[n2][n1] = g[n2][n1] - 1

    removeEdgeIfWeighIs0(g,n1,n2)
    removeEdgeIfWeighIs0(g,n2,n1)

def addEdge( g, n1,n2, weight):
    if n1 in g:
        if n2 in g[n1]:
            g[n1][n2] = g[n1][n2] + weight
        else:
            g[n1][n2] = weight
    else:
        g[n1] = {n2:weight}

def addUndirectedEdge( g, n1,n2, weight):
    addEdge(g,n1,n2,weight)
    addEdge(g,n2,n1,weight)

def orcircleDist( p1,p2, nbPillars):
    if p2 - p1 >= 0:
        return p2-p1
    else:
        return nbPillars+p2-p1

def generateGraphTrajectory( graph, nbPillar, rc,rp ):
    out = [np.array([rc, 0.0, 0.0]), np.array([rc - rp, 0.0, 0.0])]
    #tant qu'il y a des arretes
    d2 = copy.deepcopy(graph)
    cur = 0
    while len(d2) > 0:
        prevcur = cur
        keys = np.array( d2.keys() )
        dist = np.array( [ orcircleDist(prevcur,p,nbPillar) for p in keys ] )
        nextind = np.argmin(dist)
        cur = keys[nextind]
        out.extend( jumpPillar(prevcur,cur,nbPillar,rc,rp,1.0))
        print "exterior move to : " + str(cur)
        while cur in d2:
            print cur
            nextcur = d2[cur].iterkeys().next()
            out.extend( hookPillar(nextcur,nbPillar,rc,rp))
            #print (cur,nextcur)
            decreaseWeightEdge(d2,cur,nextcur)
            cur = nextcur
        print cur

    out.extend(hookPillar(nextcur, nbPillar, rc, rp))
    out.extend(jumpPillar(cur, 0, nbPillar, rc, rp, 1))
    out.append( np.array([rc - rp, 0.0, 0.0]) )
    out.append(np.array([rc , 0.0, 0.0]))
    return out


def strictOrientedCircleDist( p1, p2, nbPillars, dir, hookPenalty):
    maxValue = 10000.0
    if( p1 % nbPillars == p2 % nbPillars ):
        return maxValue

    out = dir*(p2-p1)
    if out < 0:
        out = out + nbPillars

    if out < 0:
        print "out negative"
        print (out, p1, p2, dir)

    if p1%2 == p2%2:
        return out +hookPenalty
    return out



def nextdir(pos):
    if pos % 2 == 0:
        return 1.0
    else:
        return -1.0

def PickNode1( curPosState, d2,nbPillar,hookPenalty ):
    keys = d2.keys()
    dist = np.array([strictOrientedCircleDist(curPosState, p, 2*nbPillar,nextdir( curPosState ),hookPenalty) for p in keys])
    nextind = np.argmin(dist)
    print( str.format("next node1 : {0}  dist : {1}",keys[nextind],dist[nextind]))
    #print keys
    #print dist
    return keys[nextind]



def generateGraphTrajectory2( graph, nbPillar, rc, rp):
    out = [np.array([rc, 0.0, 0.0]), np.array([rc + rp, 0.0, 0.0])]
    # tant qu'il y a des arretes
    d2 = copy.deepcopy(graph)

    #Be careful of the constraint on nextdir which should

    curPosState = 1
    while len(d2) > 0:
        #pick node 1 -> GO TO OUT 1; GO TO IN 1 FROM current posion state
        node1 = PickNode1(curPosState,d2,nbPillar,50)
        print "node1 :" + str( node1 )
        if( node1 %2 != curPosState %2):
            #we don't have to hook
            hookpillar = (node1+1) / 2
            out.extend( generateOutMoveTrajectory((curPosState+1)/2, hookpillar , nbPillar, rc, rp, nextdir( curPosState )) )
            out.extend( generateTrajectoryBetweenPillar(hookpillar,nbPillar,rc,rp, -1.0 ))
        else:
            #we need to hook the pillar
            hookpillar = (node1+1) /2 + int( nextdir(curPosState))
            outhookpillar = (node1+1) /2
            print "outhookpillar :" + str(  outhookpillar )
            out.extend(generateOutMoveTrajectory((curPosState+1)/2, outhookpillar, nbPillar, rc, rp, nextdir( curPosState) ))
            out.extend( generateTrajectoryBetweenPillar(outhookpillar, nbPillar, rc, rp, -1.0))
            out.extend( generateTrajectoryBetweenPillar(hookpillar, nbPillar, rc, rp, 1.0))
            out.extend( generateTrajectoryBetweenPillar(outhookpillar, nbPillar, rc, rp, -1.0))

        #pick node 2 -> GO TO IN 2 ;GO TO OUT 2
        #node2 = random.choice( d2[node1].keys() )
        node2 = d2[node1].keys()[0]
        print "node2 :" + str(node2)
        decreaseWeightEdge(d2,node1,node2)
        out.extend( generateTrajectoryBetweenPillar((node2+1) / 2 , nbPillar, rc, rp, 1.0) )

        #update current position state: pos = node2   if node2 % 2 == 0 nextdir = 1.0 else nextdir = -1.0

        curPosState = node2

    out.extend( generateOutMoveTrajectory((curPosState+1)/2, 0, nbPillar, rc, rp, nextdir( curPosState ) ) )
    out.append(np.array([rc + rp, 0.0, 0.0]))
    out.append(np.array([rc, 0.0, 0.0]))

    return np.stack(out)

def lengthOfWireNeeded( traj ):
    out = 0
    for i in range( 1, len(traj) ):
        dist2 = math.pow(traj[i,0] - traj[i-1,0],2.0)+math.pow(traj[i,1] - traj[i-1,1],2.0)
        out = out + math.sqrt(dist2)

    return out

def testGraph():
    g = {}
    addUndirectedEdge(g, 0, 30, 3)
    addUndirectedEdge(g, 30, 60, 3)
    addUndirectedEdge(g, 60, 90, 3)
    addUndirectedEdge(g, 0, 60, 1)
    return g

def uppillar(p):
    return 2*p+1
def downpillar(p):
    return 2*p

def testGraph2():
    g = {}
    addUndirectedEdge(g, downpillar(0), uppillar(30), 1)
    addUndirectedEdge(g, downpillar(30), uppillar(80), 1)
    addUndirectedEdge(g, downpillar(80), uppillar(90), 1)
    addUndirectedEdge(g, downpillar(90), uppillar(140), 1)
    addUndirectedEdge(g, downpillar(140), uppillar(0), 1)
    return g


def circleDist( p1,p2,nbPillars):
    return min( abs(p1-p2), nbPillars-abs(p1-p2) )

def brokenPins():
    return {149:True,1:True,13:True,14:True, 15:True,28:True,131:True,60:True}


def loadGraphFromFile( fileName ,brokenPin,apartdist, nbEdges ):
    with open(fileName)as f:
        g = {}
        l = int( next(f) )
        coords = []
        edges = []

        for i in range(l):
            line = next(f)
            coords.append( [ float(x) for x in line.split() ])
        next(f)
        for line in f:
            s = line.split()
            spl = [int(s[0]),int(s[1]),float(s[2]) ]
            if( spl[0]/2 in brokenPin or spl[1]/2 in brokenPin):
                continue

            if( circleDist( spl[0],spl[1], l) > apartdist ):
                w = math.pow( spl[2],1.0)
                #print (spl[0],spl[1], w)
                edges.append((spl[0],spl[1],w))

        if( nbEdges > 0):
            p = np.array( [e[2] for e in edges],dtype='float32' )
            p = p / p.sum()
            sel = np.random.choice(len(edges),nbEdges,True,p)
            sampledEdges = []
            for i in sel:
                sampledEdges.append(edges[i])
        else:
            sampledEdges = edges

        print "number of edges : " + str(len(sampledEdges))
        for e in sampledEdges:
            addUndirectedEdge(g, e[0], e[1], 1)
        return (l,g)


def displayTraj(traj):
    nptraj = np.stack(traj)
    plt.plot( nptraj[:,0],nptraj[:,1])
    plt.show()

def pillar2tocart( i,n,rc ):
    tanangle = math.atan( 1.0/n)

    if i%2 == 0:
        angle = getPillarAngle(i/2,n) - tanangle
    else:
        angle = getPillarAngle(i/2,n) + tanangle

    return (rc*math.cos(angle),rc*math.sin(angle))

def displayGraph2( nbPillar,g,r, lw):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    lines = []
    c = []
    d2 = copy.deepcopy(g)
    while len(d2) > 0:
        n1 = d2.keys()[0]
        n2 = d2[n1].keys()[0]

        c1 = pillar2tocart(n1,nbPillar,r)
        c2 = pillar2tocart(n2,nbPillar,r)
        #x1.append(c1[0])
        #x2.append(c2[0])
        #y1.append(c1[1])
        #y2.append(c2[1])
        lines.append(((c1[0],-c1[1]),(c2[0],-c2[1])))
        c.append( (0,0,0,1) )
        decreaseWeightEdge(d2,n1,n2)

    #lines = plt.plot( np.stack(x1),np.stack(y1),np.stack(x2),np.stack(y2))
    #plt.setp(lines, color='white', linewidth=1.0)
    #plt.gca().set_axis_bgcolor('black')
    lc = mc.LineCollection(lines,colors=np.array(c) ,linewidths=lw)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    fig.show()
    #plt.show()




def testSeq(n,dk):
    out = []
    dict = {}
    k = 0
    while k not in dict:
        out.append( k )
        dict[k] = 1
        k = (k + dk) % n
    out.append(k)
    return out;
