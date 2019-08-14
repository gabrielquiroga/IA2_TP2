
safetyValve(sv1).
safetyValve(sv2).
safetyValve(sv3).

continuousGasEvacuation(sv1, X) :- X is 0.
continuousGasEvacuation(sv2, X) :- X is 1.
continuousGasEvacuation(sv3, X) :- X is 0.

%sensorPipesBlocked(SV) :- appropiateGasPressure(SV).
sensorPipesBlocked(sv1, X) :- appropiateGasPressure(sv1), X is 0.
sensorPipesBlocked(sv2, X) :- appropiateGasPressure(sv2), X is 0.
sensorPipesBlocked(sv3, X) :- appropiateGasPressure(sv3), X is 1.
pressure(sv1, P) :- P is 40.
pressure(sv2, P) :- P is 60.
pressure(sv3, P) :- P is 30.
preventableLeakage(sv1, X) :- effectiveSafetySpring(sv1, X1), X1 is 1, X is 1.
preventableLeakage(sv2, X) :- effectiveSafetySpring(sv2, X1), X1 is 1, X is 1.
preventableLeakage(sv3, X) :- effectiveSafetySpring(sv3, X1), X1 is 1, X is 0.
effectiveSafetySpring(sv1, X) :- sensorPipesBlocked(sv1, X1), X1 is 0, X is 0.
effectiveSafetySpring(sv2, X) :- sensorPipesBlocked(sv2, X1), X1 is 0, X is 0.
effectiveSafetySpring(sv3, X) :- sensorPipesBlocked(sv3, X1), X1 is 0, X is 1.

pilotWorksProperly(sv1, X) :- properLeakagePrevention(sv1, X1), X1 is 1, X is 0.
pilotWorksProperly(sv2, X) :- properLeakagePrevention(sv2, X1), X1 is 1, X is 1.
pilotWorksProperly(sv3, X) :- properLeakagePrevention(sv3, X1), X1 is 1, X is 1.
properLeakagePrevention(sv1, X) :- goodSpringPerformance(sv1, X1), X1 is 1, X is 1.
properLeakagePrevention(sv2, X) :- goodSpringPerformance(sv2, X1), X1 is 1, X is 1.
properLeakagePrevention(sv3, X) :- goodSpringPerformance(sv3, X1), X1 is 1, X is 0.
goodSpringPerformance(sv1, X) :- valveSensorsBlocked(sv1, X1), X1 is 0, X is 0.
goodSpringPerformance(sv2, X) :- valveSensorsBlocked(sv2, X1), X1 is 0, X is 1.
goodSpringPerformance(sv3, X) :- valveSensorsBlocked(sv3, X1), X1 is 0, X is 1.
valveSensorsBlocked(sv1, X) :- valveStatusClosed(sv1, X1), X1 is 0, X is 0.
valveSensorsBlocked(sv2, X) :- valveStatusClosed(sv2, X1), X1 is 0, X is 0.
valveSensorsBlocked(sv3, X) :- valveStatusClosed(sv3, X1), X1 is 0, X is 1.
valveStatusClosed(sv1, X) :- reliefValveWorksProperly(sv1, X1), X1 is 0, X is 0.
valveStatusClosed(sv2, X) :- reliefValveWorksProperly(sv2, X1), X1 is 0, X is 1.
valveStatusClosed(sv3, X) :- reliefValveWorksProperly(sv3, X1), X1 is 0, X is 1.
reliefValveWorksProperly(sv1, X) :- continuousGasEvacuation(sv1, X1), X1 is 0, X is 0.
reliefValveWorksProperly(sv2, X) :- continuousGasEvacuation(sv2, X1), X1 is 0, X is 0.
reliefValveWorksProperly(sv3, X) :- continuousGasEvacuation(sv3, X1), X1 is 0, X is 1.




%RAMA CENTRAL-DERECHA
cleanAndFix(SV) :- safetyValve(SV), sensorPipesBlocked(SV, X), X is 1.

adjustPressure(SV) :- safetyValve(SV), not(appropiateGasPressure(SV)), continuousGasEvacuation(SV, X), X is 1.
appropiateGasPressure(SV) :- safetyValve(SV), pressure(SV, P), P < 50, continuousGasEvacuation(SV, X), X is 1.

setSafetyValve(SV) :- safetyValve(SV), preventableLeakage(SV, X), X is 1.
replaceSitAndOrifice(SV) :- safetyValve(SV), preventableLeakage(SV, X), X is 0.

replaceSafetySpring(SV) :- safetyValve(SV), effectiveSafetySpring(SV, X), X is 0.

%RAMA CENTRAL-IZQUIERDA
setSafetyValve(SV) :- safetyValve(SV), pilotWorksProperly(SV, X), X is 1.
pilotServiceAndReinstallation(SV) :- safetyValve(SV), pilotWorksProperly(SV, X), X is 0.

replaceSitAndOrifice(SV) :- safetyValve(SV), properLeakagePrevention(SV, X), X is 0.

serviceSafetySpring(SV) :- safetyValve(SV), goodSpringPerformance(SV, X), X is 0.

cleanAndTroubleshoot(SV) :- safetyValve(SV), valveSensorsBlocked(SV, X), X is 1.

setOpenPosition(SV) :- safetyValve(SV), valveStatusClosed(SV, X), X is 1.

appropiateSafetyFunction(SV) :- safetyValve(SV), reliefValveWorksProperly(SV, X), X is 1.

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