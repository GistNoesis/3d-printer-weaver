

module ring( r,h,w )
{
difference()
{
    ir = r-w/2.0;
    or = r+w/2.0;
    cylinder( h= h, r1=or, r2=or,center=false,$fn=200);
    translate([0,0,-0.5]) cylinder( h=h+1,r1=ir,r2=ir,center=false,$fn=200);
}
}


module pillars( r, h, nb, offset = 0.5 )
{
    dtheta = 360.0 / nb; 
    for ( i = [0:nb] )
    {
        thetai = dtheta * (i+offset) ;
        x = r * cos(thetai);
        y = r * sin(thetai);
        rmid = 1.1;
        translate( [x,y,h/2]) 
            cylinder( h=h, r1 = 1.0, r2=rmid, center = true, $fn=10);
    }
    
}

r = 110;

ring( r,2,5);

translate( [0,0,2] ) pillars( r, 2, 150,0.5);
//translate( [0,0,2] ) pillars( r-1.5, 2, 75,0.0);
//translate( [0,0,2] ) pillars( r+1.5, 2, 75,0.5);
