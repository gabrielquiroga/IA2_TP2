
safetyValve(sv1).
safetyValve(sv2).
safetyValve(sv3).

continuousGasEvacuation(sv1, 0).
continuousGasEvacuation(sv2, 1).
continuousGasEvacuation(sv3, 0).
cge(SV) :- safetyValve(SV),  continuousGasEvacuation(SV, 1).

%sensorPipesBlocked(SV) :- appropiateGasPressure(SV).
:- dynamic continuousGasEvacuation/2.
:- dynamic sensorPipesBlocked/2.
:- dynamic pressure/2.
:- dynamic preventableLeakage/2.
:- dynamic effectiveSafetySpring/2.
:- dynamic pilotWorksProperly/2.
:- dynamic properLeakagePrevention/2.
:- dynamic goodSpringPerformance/2.
:- dynamic valveSensorsBlocked/2.
:- dynamic valveStatusClosed/2.
:- dynamic reliefValveWorksProperly/2.
sensorPipesBlocked(sv1, 0).
sensorPipesBlocked(sv2, 0).
sensorPipesBlocked(sv3, 1).
spb(SV) :- safetyValve(SV), sensorPipesBlocked(SV, 1).
agp(SV) :- safetyValve(SV), pressure(SV, P), P < 50.
pressure(sv1, 40).
pressure(sv2, 60).
pressure(sv3, 30).
preventableLeakage(sv1, 1).
preventableLeakage(sv2, 1).
preventableLeakage(sv3, 0).
pl(SV) :- safetyValve(SV), preventableLeakage(SV, 1).
effectiveSafetySpring(sv1, 0).
effectiveSafetySpring(sv2, 0).
effectiveSafetySpring(sv3, 1).
esp(SV) :- safetyValve(SV), effectiveSafetySpring(SV, 1).

pilotWorksProperly(sv1, 0).
pilotWorksProperly(sv2, 1).
pilotWorksProperly(sv3, 1).
pwp(SV) :- safetyValve(SV), pilotWorksProperly(SV, 1).
properLeakagePrevention(sv1, 1).
properLeakagePrevention(sv2, 1).
properLeakagePrevention(sv3, 0).
plp(SV) :- safetyValve(SV), properLeakagePrevention(SV, 1).
goodSpringPerformance(sv1, 0).
goodSpringPerformance(sv2, 1).
goodSpringPerformance(sv3, 1).
gsp(SV) :- safetyValve(SV), goodSpringPerformance(SV, 1).
valveSensorsBlocked(sv1, 0).
valveSensorsBlocked(sv2, 0).
valveSensorsBlocked(sv3, 1).
vsb(SV) :- safetyValve(SV), valveSensorsBlocked(SV, 1).
valveStatusClosed(sv1, 0).
valveStatusClosed(sv2, 1).
valveStatusClosed(sv3, 1).
vsc(SV) :- safetyValve(SV), valveStatusClosed(SV, 1).
reliefValveWorksProperly(sv1, 0).
reliefValveWorksProperly(sv2, 0).
reliefValveWorksProperly(sv3, 1).
rvwp(SV) :- safetyValve(SV), reliefValveWorksProperly(SV, 1).


%RAMA CENTRAL-DERECHA
cleanAndFix(SV) :- safetyValve(SV), spb(SV), agp(SV), cge(SV).

adjustPressure(SV) :- safetyValve(SV), not(agp(SV)), cge(SV).

setSafetyValve(SV) :- safetyValve(SV), pl(SV), esp(SV), not(spb(SV)), agp(SV), cge(SV).
replaceSitAndOrifice(SV) :- safetyValve(SV), not(pl(SV)), esp(SV), not(spb(SV)), agp(SV), cge(SV).

replaceSafetySpring(SV) :- safetyValve(SV), not(esp(SV)), not(spb(SV)), agp(SV), cge(SV).

%RAMA CENTRAL-IZQUIERDA
setSafetyValve(SV) :- safetyValve(SV), pwp(SV), plp(SV), gsp(SV), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).
pilotServiceAndReinstallation(SV) :- safetyValve(SV), not(pwp(SV)), plp(SV), gsp(SV), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

replaceSitAndOrifice(SV) :- safetyValve(SV), not(plp(SV)), gsp(SV), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

serviceSafetySpring(SV) :- safetyValve(SV), not(gsp(SV)), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

cleanAndTroubleshoot(SV) :- safetyValve(SV), vsb(SV), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

setOpenPosition(SV) :- safetyValve(SV), vsc(SV), not(rvwp(SV)), not(cge(SV)).

appropiateSafetyFunction(SV) :- safetyValve(SV), rvwp(SV), not(cge(SV)).

%Para preguntas abiertas
accion(cleanAndFix).
accion(adjustPressure).
accion(setSafetyValve).
accion(replaceSitAndOrifice).
accion(replaceSafetySpring).
accion(pilotServiceAndReinstallation).
accion(serviceSafetySpring).
accion(cleanAndTroubleshoot).
accion(setOpenPosition).
accion(appropiateSafetyFunction).
actionToDo(cleanAndFix, SV) :- safetyValve(SV), cleanAndFix(SV).
actionToDo(adjustPressure, SV) :- safetyValve(SV), adjustPressure(SV).
actionToDo(setSafetyValve, SV) :- safetyValve(SV), setSafetyValve(SV).
actionToDo(replaceSitAndOrifice, SV) :- safetyValve(SV), replaceSitAndOrifice(SV).
actionToDo(replaceSafetySpring, SV) :- safetyValve(SV), replaceSafetySpring(SV).
actionToDo(pilotServiceAndReinstallation, SV) :- safetyValve(SV), pilotServiceAndReinstallation(SV).
actionToDo(serviceSafetySpring, SV) :- safetyValve(SV), serviceSafetySpring(SV).
actionToDo(cleanAndTroubleshoot, SV) :- safetyValve(SV), cleanAndTroubleshoot(SV).
actionToDo(setOpenPosition, SV) :- safetyValve(SV), setOpenPosition(SV).
actionToDo(appropiateSafetyFunction, SV) :- safetyValve(SV), appropiateSafetyFunction(SV).