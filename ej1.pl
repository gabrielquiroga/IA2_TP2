%base de conocimientos

%rama izquierda

parts(body).
parts(pipes).
parts(joints).

thickness_treshold(body) :- thickness(body,T),T > 5.
thickness_treshold(pipes) :- thickness(pipes,T),T > 4.
thickness_treshold(joints) :- thickness(joints,T),T > 3.


renderandpaint(P) :- parts(P),rusting(P,true);dazzling(P,true). %si es true significa que hay que pintar osea que es malo 
%rama derecha
soap_test(joints):- gas_leakage(joints,L),not(L=true). %pasa el soap test si no hay derrames
leakage_solution(joints):- wrench_fix(joints,true),writeln('report joints to technical inspection unit');wrench_fix(joints,false),writeln('report joints to repair department to fix the fault').

%rama central
safetyValve(sv1).
safetyValve(sv2).
safetyValve(sv3).

cge(SV) :- safetyValve(SV),  continuousGasEvacuation(SV, true).
spb(SV) :- safetyValve(SV), sensorPipesBlocked(SV, true).
agp(SV) :- safetyValve(SV), pressure(SV, P), P < 50.
pl(SV) :- safetyValve(SV), preventableLeakage(SV, true).
esp(SV) :- safetyValve(SV), effectiveSafetySpring(SV, true).
pwp(SV) :- safetyValve(SV), pilotWorksProperly(SV, true).
plp(SV) :- safetyValve(SV), properLeakagePrevention(SV, true).
gsp(SV) :- safetyValve(SV), goodSpringPerformance(SV, true).
vsb(SV) :- safetyValve(SV), valveSensorsBlocked(SV, true).
vsc(SV) :- safetyValve(SV), valveStatusClosed(SV, true).
rvwp(SV) :- safetyValve(SV), reliefValveWorksProperly(SV, true).

%RAMA CENTRAL-DERECHA
cleanAndFix(SV) :- safetyValve(SV), spb(SV), agp(SV), cge(SV).

adjustPressure(SV) :- safetyValve(SV), not(agp(SV)), cge(SV).

setSafetyValve(SV) :- safetyValve(SV), pl(SV), esp(SV), not(spb(SV)), agp(SV), cge(SV).
replaceSitAndOrifice(SV) :- safetyValve(SV), not(pl(SV)), esp(SV), not(spb(SV)), agp(SV), cge(SV).

replaceSafetySpring(SV) :- safetyValve(SV), not(esp(SV)), not(spb(SV)), agp(SV), cge(SV).

%RAMA CENTRAL-IZQUIERDA
setSafetyValveIzq(SV) :- safetyValve(SV), pwp(SV), plp(SV), gsp(SV), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).
pilotServiceAndReinstallation(SV) :- safetyValve(SV), not(pwp(SV)), plp(SV), gsp(SV), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

replaceSitAndOrificeIzq(SV) :- safetyValve(SV), not(plp(SV)), gsp(SV), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

serviceSafetySpring(SV) :- safetyValve(SV), not(gsp(SV)), not(vsb(SV)), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

cleanAndTroubleshoot(SV) :- safetyValve(SV), vsb(SV), not(vsc(SV)), not(rvwp(SV)), not(cge(SV)).

setOpenPosition(SV) :- safetyValve(SV), vsc(SV), not(rvwp(SV)), not(cge(SV)).

appropiateSafetyFunction(SV) :- safetyValve(SV), rvwp(SV), not(cge(SV)).

%Gran afirmacion que revisa todas las condiciones
safety(estacion) :- not(renderandpaint(body)),not(renderandpaint(joints)),not(renderandpaint(pipes)),
                thickness_treshold(body),thickness_treshold(pipes),thickness_treshold(joints),
                soap_test(joints),
                not(cleanAndFix(sv1)),not(cleanAndFix(sv2)),not(cleanAndFix(sv3)),
                not(adjustPressure(sv1)),not(adjustPressure(sv2)),not(adjustPressure(sv3)),
                not(setSafetyValve(sv1)),not(setSafetyValve(sv2)),not(setSafetyValve(sv3)),
                not(replaceSitAndOrifice(sv1)),not(replaceSitAndOrifice(sv2)),not(replaceSitAndOrifice(sv3)),  
                not(replaceSafetySpring(sv1)),not(replaceSafetySpring(sv2)),not(replaceSafetySpring(sv3)),
                not(pilotServiceAndReinstallation(sv1)),not(pilotServiceAndReinstallation(sv2)),not(pilotServiceAndReinstallation(sv3)),
                not(setSafetyValveIzq(sv1)),not(setSafetyValveIzq(sv2)),not(setSafetyValveIzq(sv3)),
                not(serviceSafetySpring(sv1)),not(serviceSafetySpring(sv2)),not(serviceSafetySpring(sv3)),
                not(replaceSitAndOrificeIzq(sv1)),not(replaceSitAndOrificeIzq(sv2)),not(replaceSitAndOrificeIzq(sv3)),
                not(cleanAndTroubleshoot(sv1)),not(cleanAndTroubleshoot(sv2)),not(cleanAndTroubleshoot(sv3)),
                not(setOpenPosition(sv1)),not(setOpenPosition(sv2)),not(setOpenPosition(sv3)),
                appropiateSafetyFunction(sv1),appropiateSafetyFunction(sv2),appropiateSafetyFunction(sv3).
quehago(estacion):- renderandpaint(body),writeln('render and paint the body');renderandpaint(joints),writeln('render and paint the joints');renderandpaint(pipes),writeln('render and paint the pipes');
                not(thickness_treshold(body)),writeln('report the body to the Technical Inspection Unit: Thickness problem');not(thickness_treshold(pipes)),writeln('report the pipes to the Technical Inspection Unit: Thickness problem');not(thickness_treshold(joints)),writeln('report the joints to the Technical Inspection Unit: Thickness problem');
                not(soap_test(joints)),leakage_solution(joints);
                cleanAndFix(sv1),writeln('Clean And Fix the faults of the sensing pipes of sv1');cleanAndFix(sv2),writeln('Clean And Fix the faults of the sensing pipes of sv2');cleanAndFix(sv3),writeln('Clean And Fix the faults of the sensing pipes of sv3');
                adjustPressure(sv1),writeln('Adjust the regulator of the line gas pressure of sv1');adjustPressure(sv2),writeln('Adjust the regulator of the line gas pressure of sv2');adjustPressure(sv3),writeln('Adjust the regulator of the line gas pressure of sv3');
                setSafetyValve(sv1),writeln('Preventable leakage between sit and orifice in sv1, set safety valve according to instructions');setSafetyValve(sv2),writeln('Preventable leakage between sit and orifice in sv2, set safety valve according to instructions');setSafetyValve(sv3),writeln('Preventable leakage between sit and orifice in sv3, set safety valve according to instructions');
                replaceSitAndOrifice(sv1),writeln('Replace Sit and orifice and put sv1 into circuit');replaceSitAndOrifice(sv2),writeln('Replace Sit and orifice and put sv2 into circuit');replaceSitAndOrifice(sv3),writeln('Replace Sit and orifice and put sv3 into circuit');
                replaceSafetySpring(sv1),writeln('Safety spring in sv1 ineffective, replace it');replaceSafetySpring(sv2),writeln('Safety spring in sv2 ineffective, replace it');replaceSafetySpring(sv3),writeln('Safety spring in sv3 ineffective, replace it');
                pilotServiceAndReinstallation(sv1),writeln('Pilot in sv1 not working properly, full service and reinstallation required');pilotServiceAndReinstallation(sv2),writeln('Pilot in sv2 not working properly, full service and reinstallation required');pilotServiceAndReinstallation(sv3),writeln('Pilot in sv3 not working properly, full service and reinstallation required');
                setSafetyValveIzq(sv1),writeln('Pilot in sv1 working properly, set safety valve according to the instrucions');setSafetyValveIzq(sv2),writeln('Pilot in sv2 working properly, set safety valve according to the instrucions'); setSafetyValveIzq(sv3),writeln('Pilot in sv3 working properly, set safety valve according to the instrucions'); 
                serviceSafetySpring(sv1),writeln('Put safety spring of sv1 into service');serviceSafetySpring(sv2),writeln('Put safety spring of sv2 into service');serviceSafetySpring(sv3),writeln('Put safety spring of sv3 into service');
                replaceSitAndOrificeIzq(sv1),writeln('No proper leakage prevention between sit and orifice, replace Sit and orifice and put sv1 into circuit');replaceSitAndOrificeIzq(sv2),writeln('No proper leakage prevention between sit and orifice, replace Sit and orifice and put sv2 into circuit');replaceSitAndOrificeIzq(sv3),writeln('No proper leakage prevention between sit and orifice, replace Sit and orifice and put sv3 into circuit');
                cleanAndTroubleshoot(sv1),writeln('Control valve sensors blocked, clean and troubleshoot sv1  sensing pipes');cleanAndTroubleshoot(sv2),writeln('Control valve sensors blocked, clean and troubleshoot sv2  sensing pipes');cleanAndTroubleshoot(sv3),writeln('Control valve sensors blocked, clean and troubleshoot sv3  sensing pipes');
                setOpenPosition(sv1),writeln('Place sv1 in open position');setOpenPosition(sv2),writeln('Place sv2 in open position');setOpenPosition(sv3),writeln('Place sv3 in open position').
%Para preguntas abiertas
actionToDo(thickness_treshold,P):-parts(P),thickness_treshold(P).
actionToDo(renderandpaint,P):-parts(P),renderandpaint(P).
actionToDo(soap_test, joints):-soap_test(joints).
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
actionToDo(replaceSitAndOrificeIzq, SV) :- safetyValve(SV), replaceSitAndOrificeIzq(SV).
actionToDo(setSafetyValveIzq, SV) :- safetyValve(SV), setSafetyValveIzq(SV).

%Ground Facts

%rama izquierda
:- dynamic thickness/2.
thickness(body, 7).
thickness(pipes, 6).
thickness(joints, 5).

:- dynamic rusting/2.
rusting(body,false).
rusting(pipes,false).
rusting(joints,false).

:- dynamic dazzling/2.
dazzling(body, false).
dazzling(pipes,false).
dazzling(joints, false).

%rama derecha
:- dynamic gas_leakage/2.
gas_leakage(joints, false). 

:- dynamic wrench_fix/2.
wrench_fix(joints,false). 

%rama central

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

continuousGasEvacuation(sv1, false).
continuousGasEvacuation(sv2, false).
continuousGasEvacuation(sv3, false).

sensorPipesBlocked(sv1, false).
sensorPipesBlocked(sv2, false).
sensorPipesBlocked(sv3, false).

pressure(sv1, 40).
pressure(sv2, 60).
pressure(sv3, 30).

preventableLeakage(sv1, true).
preventableLeakage(sv2, true).
preventableLeakage(sv3, true).

effectiveSafetySpring(sv1, false).
effectiveSafetySpring(sv2, false).
effectiveSafetySpring(sv3, false).

pilotWorksProperly(sv1, true).
pilotWorksProperly(sv2, true).
pilotWorksProperly(sv3, true).

properLeakagePrevention(sv1, true).
properLeakagePrevention(sv2, true).
properLeakagePrevention(sv3,true).

goodSpringPerformance(sv1, true).
goodSpringPerformance(sv2, true).
goodSpringPerformance(sv3, true).

valveSensorsBlocked(sv1, false).
valveSensorsBlocked(sv2, false).
valveSensorsBlocked(sv3, false).

valveStatusClosed(sv1, false).
valveStatusClosed(sv2, false).
valveStatusClosed(sv3, false).

reliefValveWorksProperly(sv1, true).
reliefValveWorksProperly(sv2, true).
reliefValveWorksProperly(sv3, true).
