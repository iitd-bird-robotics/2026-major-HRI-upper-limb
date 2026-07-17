import opensim as osim
def addBodies(model, External_load):
    load2 = osim.Body("load2", External_load, osim.Vec3(0),osim.Inertia(0.000001))
    cyl2 = osim.Mesh('External_Load.obj')
    cyl2.setColor(osim.Vec3(0))
    load2.attachGeometry(cyl2)
    model.addBody(load2)
    load_fix2 = osim.WeldJoint('load_fix2',model.getBodySet().get(1),osim.Vec3(0.05,-0.35,0.1),osim.Vec3(0),model.getBodySet().get(2),osim.Vec3(0),osim.Vec3(0))
    model.addJoint(load_fix2)
    armStrap = osim.Body('armStrap',0.003, osim.Vec3(0),osim.Inertia(0.000001))
    cyl2 = osim.Mesh('Arm_Strap.obj')                                               # Importing the geometry file, to be defined as the geometry of the arm strap
    cyl2.setColor(osim.Vec3(0.4,0.4,0.4))                                           # Setting the colour of the new geometry
    armStrap.attachGeometry(cyl2)                                             # Attching the geometry to the Body 'armStrap'
    model.addBody(armStrap)    
    armStrapJ = osim.WeldJoint('armStrapJ',model.getBodySet().get(0),osim.Vec3(0.0,-0.14,0.0),osim.Vec3(0),model.getBodySet().get(3),osim.Vec3(0),osim.Vec3(0))
    model.addJoint(armStrapJ)
    forearmStrap = osim.Body('forearmStrap',0.003, osim.Vec3(0),osim.Inertia(0.000001))            #Defining the mass properties of the strap
    cyl2 = osim.Mesh('Forearm_Strap.obj')                                           #Importing the geometry file, to be defined as the geometry of the forearm strap
    cyl2.setColor(osim.Vec3(0.4,0.4,0.4))                                           #Setting the colour of the new geometry
    forearmStrap.attachGeometry(cyl2)                                          # Attching the geometry to the Body 'forearmStrap'
    forearmStrap.scaleAttachedGeometry(osim.Vec3(1.2,2,1.5))
    model.addBody(forearmStrap)
    forearmStrapJ = osim.WeldJoint('forearmStrapJ',model.getBodySet().get(1),osim.Vec3(0.0,-0.15,0.05),osim.Vec3(0),model.getBodySet().get(4),osim.Vec3(0),osim.Vec3(0.1,0,0))
    model.addJoint(forearmStrapJ)
    return model;