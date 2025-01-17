import numpy as np

def eps_less(v1,v2):
    return np.logical_and(~np.isclose(v1,v2), v1<v2)

def eps_less_eq(v1,v2):
    return np.logical_or(np.isclose(v1,v2), v1<v2)

def line_intersect2(p0,p1,q0,q1):
    '''
    judge if line (p0,p1) intersects with line(q0,q1)
    '''
    vp = p1-p0
    vq = q1-q0
    vx = p0-q0
    d = np.cross(vp, vq)
    u = np.cross(vq, vx)
    v = np.cross(vp, vx)
    mask = d < 0
    u[mask] = -u[mask]
    v[mask] = -v[mask]
    d[mask] = -d[mask]
    return np.logical_and.reduce((eps_less(0,u), eps_less(0,v), eps_less(u,d), eps_less(v,d)))

def point_in_triangle2(a,b,c,p):
    ac = c-a
    ab = b-a
    ap = p-a
    u = np.cross(ap,ac)
    v = np.cross(ab,ap)
    d = np.cross(ab,ac)
    mask = d < 0
    mask_ = np.logical_or(mask, np.zeros(u.shape))
    u[mask_] = -u[mask_]
    v[mask_] = -v[mask_]
    d[mask] = -d[mask]
    return np.logical_and.reduce((eps_less(0,u), eps_less(0,v), eps_less(u+v, d)))

def point_in_triangle3(a,b,c,p):
    ac = c-a
    ab = b-a
    ap = p-a
    u = np.cross(ap,ac)
    v = np.cross(ab,ap)
    d = np.cross(ab,ac)
    mask = d < 0
    mask_ = np.logical_or(mask, np.zeros(u.shape))
    u[mask_] = -u[mask_]
    v[mask_] = -v[mask_]
    d[mask] = -d[mask]
    return np.logical_and.reduce((eps_less_eq(0,u), eps_less_eq(0,v), eps_less_eq(u+v, d)))

def tri_intersect2(t1, t2):
    '''
    judge if two triangles in a plane intersect 

    '''
    a=np.any(line_intersect2(t1[:,None,[0,1,2],None,:],t1[:,None,[1,2,0],None,:],t2[None,:,None,[0,1,2],:],t2[None,:,None,[1,2,0],:]), axis=(2,3))
    b=np.any(point_in_triangle2(t1[:,None,0,None,:],t1[:,None,1,None,:],t1[:,None,2,None,:],t2[None,:,:,:]),axis=2)
    c=np.any(point_in_triangle2(t2[None,:,0,None,:],t2[None,:,1,None,:],t2[None,:,2,None,:],t1[:,None,:,:]),axis=2)
    d=np.all(point_in_triangle3(t1[:,None,0,None,:],t1[:,None,1,None,:],t1[:,None,2,None,:],t2[None,:,:,:]),axis=2)
    e=np.all(point_in_triangle3(t2[None,:,0,None,:],t2[None,:,1,None,:],t2[None,:,2,None,:],t1[:,None,:,:]),axis=2)
    return np.logical_or.reduce((a,b,c,d,e))
