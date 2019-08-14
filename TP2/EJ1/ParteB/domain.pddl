;Header and description

(define (domain acciones_de_mecanizado)

;remove requirements that are not needed
(:requirements :strips :fluents :typing :conditional-effects :negative-preconditions :equality)

(:types Tool Piece Angle Distance
;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
)

; un-comment following line if constants are needed
;(:constants )

(:predicates (Tool ?t) (Piece ?p) (Distance ?d) (Angle ?a)
            (cambiar_herramienta ?t1 ?t2) (girar_pieza ?p ?a1 ?a2) (mover_pieza ?p ?d1 ?d2) (using ?t)
            (agujerear_pieza ?p ?d1 ?a1) (desbastar_pieza ?p ?d1 ?a1) (tornear_pieza ?p ?d1)
            (in ?p ?d1) (at ?p ?a1) (agujero_en ?p ?d1 ?a1) (desbaste_en ?p ?d1 ?a1) (torneado_en ?p ?d1)
;todo: define predicates here
)

;(:functions ;todo: define numeric functions here
;)

;define actions here
(:action cambiar_herramienta
    :parameters (?t1 - Tool ?t2 - Tool)
    :precondition (and (Tool ?t1) (Tool ?t2) (using ?t1))
    :effect (and (not (using ?t1)) (using ?t2))
)
(:action girar_pieza
    :parameters (?p - Piece ?a1 - Angle ?a2 - Angle)
    :precondition (and (Piece ?p) (Angle ?a1) (Angle ?a2) (at ?p ?a1))
    :effect (and (not(at ?p ?a1)) (at ?p ?a2))
)
(:action mover_pieza
    :parameters (?p - Piece ?d1 - Distance ?d2 - Distance)
    :precondition (and (Piece ?p) (Distance ?d1) (Distance ?d2) (in ?p ?d1))
    :effect (and (not(in ?p ?d1)) (in ?p ?d2))
)
(:action agujerear_pieza
    :parameters (?p - Piece ?d1 - Distance ?a1 - Angle)
    :precondition (and (Piece ?p) (Distance ?d1) (Angle ?a1) (in ?p ?d1) (at ?p ?a1) (using Taladro))
    :effect (and (agujero_en ?p ?d1 ?a1))
)
(:action desbastar_pieza
    :parameters (?p - Piece ?d1 - Distance ?a1 - Angle)
    :precondition (and (Piece ?p) (Distance ?d1) (Angle ?a1) (in ?p ?d1) (at ?p ?a1) (using Fresa))
    :effect (and (desbaste_en ?p ?d1 ?a1))
)
(:action tornear_pieza
    :parameters (?p - Piece ?d1 - Distance)
    :precondition (and (Piece ?p) (Distance ?d1) (in ?p ?d1) (using Taladro))
    :effect (and (torneado_en ?p ?d1))
)


)