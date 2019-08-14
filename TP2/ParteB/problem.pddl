(define (problem mecanizado_de_pieza1) (:domain acciones_de_mecanizado)
(:objects Pieza - Piece
        Fresa Taladro Torno - Tool
        d0 d1 d2 d3 d4 d5 - Distance
        a0 a1 a2 a3 - Angle
)

(:init 
    (Piece Pieza)
    (Tool Fresa) (Tool Taladro) (Tool Torno)
    (Distance d0) (Distance d1) (Distance d2) (Distance d3) (Distance d4) (Distance d5)
    (Angle a0) (Angle a1) (Angle a2) (Angle a3)
    (in Pieza d0) (at Pieza a0) (using Torno)
    ;todo: put the initial state's facts and numeric values here
)

(:goal (and
    (agujero_en Pieza d1 a1)
    (desbaste_en Pieza d3 a3)
    (desbaste_en Pieza d2 a2) 
    (torneado_en Pieza d0)
    (torneado_en Pieza d4) 
    )
    ;todo: put the goal condition here
)

;un-comment the following line if metric is needed
;(:metric minimize (???))
)