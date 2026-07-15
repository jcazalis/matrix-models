(* ::Package:: *)

(* ::Title:: *)
(*Operator Testing Package*)


(* ::Author:: *)
(*Author: Jean Cazalis*)


(* ::Affiliation:: *)
(*Affiliation: DESY, CQTA*)


(* ::Abstract:: *)
(*This  package  provides  automated  testing  utilities  for  verifying  operator  equalities . It  handles  both  explicit  index  notation  (e . g ., a[1, 2])  and  implicit  Einstein  summation  notation  (e . g ., a[k1, k2]**ad[k2, k1]) . The  package  uses  the  LadderAlgebra  framework  to  simplify  and  verify  operator  identities .*)


(* ::Chapter:: *)
(*Package Public Interface*)


BeginPackage["OperatorTesting`", {"LadderAlgebra`"}];

(* Testing functions *)
TestOperatorEquality::usage = 
  "TestOperatorEquality[lhs, rhs] tests if two operator expressions are equal. \
Automatically detects whether indices are explicit or implicit.";

TestOperatorEqualityExplicit::usage = 
  "TestOperatorEqualityExplicit[lhs, rhs] tests equality for expressions with explicit indices \
using anti-normal ordering and simplification.";

TestOperatorEqualityImplicit::usage = 
  "TestOperatorEqualityImplicit[lhs, rhs, maxDim] tests equality for expressions with implicit indices \
(Einstein notation) by expanding for dimensions 1 to maxDim.";

ExpandImplicitIndices::usage = 
  "ExpandImplicitIndices[expr, dim] expands an expression with implicit indices (k1, k2, etc.) \
into explicit summation over indices from 1 to dim.";

GetImplicitIndices::usage = 
  "GetImplicitIndices[expr] returns a list of unique implicit indices in an expression.";

IsExplicitIndices::usage = 
  "IsExplicitIndices[expr] returns True if all indices in the expression are numeric.";

TestOnFockStates::usage = 
  "TestOnFockStates[lhs, rhs, dim, Lambda] tests operator equality by evaluating \
on all Fock states with dimension dim and excitations up to Lambda.";

TestMultipleEqualities::usage = 
  "TestMultipleEqualities[{{lhs1, rhs1}, {lhs2, rhs2}, ...}, maxDim] tests \
multiple operator equalities and returns a summary.";


(* ::Chapter:: *)
(*Package Implementation*)


Begin["`Private`"];


(* ::Section:: *)
(*Index Detection and Manipulation*)


(* ::Subsection:: *)
(*Identify Index Types*)


(* Check if an index is numeric (explicit) *)
NumericIndexQ[idx_] := NumericQ[idx] && IntegerQ[idx];

(* Get all indices from operators in an expression *)
GetAllIndices[expr_] := 
  Module[{operators, indices},
    (* Extract all a and ad operators *)
    operators = Cases[expr, (a|ad|\[Delta])[_, _], {0, Infinity}];
    (* Extract their indices *)
    indices = Flatten[List @@@ operators];
    DeleteDuplicates[indices]
  ];

(* Check if all indices are explicit (numeric) *)
IsExplicitIndices[expr_] := 
  Module[{indices},
    indices = GetAllIndices[expr];
    If[Length[indices] == 0, 
      True,  (* No indices means it's a constant *)
      AllTrue[indices, NumericIndexQ]
    ]
  ];

(* Get implicit indices (symbols like k1, k2, i1, j1, etc.) *)
GetImplicitIndices[expr_] := 
  Module[{allIndices},
    allIndices = GetAllIndices[expr];
    (* Select non-numeric indices *)
    Select[allIndices, !NumericIndexQ[#]&]
  ];


(* ::Section:: *)
(*Index Expansion for Einstein Notation*)


(* ::Subsection:: *)
(*Expand Single Term*)


(* Expand a single term with implicit indices *)
ExpandTermByIndices[term_, dim_Integer] :=
  Module[{implicitIndices, iterators},
    implicitIndices = GetImplicitIndices[term];
    
    If[Length[implicitIndices] == 0,
      (* No implicit indices, return as is *)
      Return[term],
      (* Create iterators for each implicit index *)
      iterators = {#, 1, dim}& /@ implicitIndices;
      (* Sum over all combinations *)
      Sum[term, Evaluate[Sequence @@ iterators]]
    ]
  ];


(* ::Subsection:: *)
(*Expand Full Expression*)


ExpandImplicitIndices[expr_, dim_Integer] :=
  Module[{terms},
    (* Expand and separate into terms *)
    terms = If[Head[Expand[expr]] === Plus,
      List @@ Expand[expr],
      {Expand[expr]}
    ];
    
    (* Expand each term and sum *)
    Total[ExpandTermByIndices[#, dim]& /@ terms]
  ];


(* ::Section:: *)
(*Testing Functions*)


(* ::Subsection:: *)
(*Test with Explicit Indices*)


TestOperatorEqualityExplicit[lhs_, rhs_] :=
  Module[{simplifiedLHS, simplifiedRHS, difference},
    (* Apply anti-normal ordering and simplification *)
    simplifiedLHS = SimplifyNC[lhs];
    simplifiedRHS = SimplifyNC[rhs];
    
    (* Check if the difference simplifies to zero *)
    difference = Simplify[simplifiedLHS - simplifiedRHS];
    
    (* Return result *)
    If[difference === 0,
      True,
      Echo[StringJoin[{"Difference after simplification: ", ToString[difference]}]];
      False
    ]
  ];


(* ::Subsection:: *)
(*Test with Implicit Indices*)


TestOperatorEqualityImplicit[lhs_, rhs_, maxDim_Integer] :=
  Module[{results, dim},
    results = Table[
      Module[{expandedLHS, expandedRHS, testResult},
        (* Expand both sides for this dimension *)
        expandedLHS = ExpandImplicitIndices[lhs, dim]/.d->dim;
        expandedRHS = ExpandImplicitIndices[rhs, dim]/.d->dim;
        
        (* Test equality *)
        testResult = TestOperatorEqualityExplicit[expandedLHS, expandedRHS];
        
        (* Print intermediate result *)
        If[!testResult,
          Print["Test failed for dim = ", dim]
        ];
        
        testResult
      ],
      {dim, 1, maxDim}
    ];
    
    (* Return True only if all tests passed *)
    If[AllTrue[results, #&],
      Echo[StringJoin[{"All tests passed for dimensions 1 to ", ToString[maxDim]}]];
      True,
      Echo["Some tests failed"];
      False
    ]
  ];


(* ::Subsection:: *)
(*Automatic Detection and Testing*)


TestOperatorEquality[lhs_, rhs_, maxDim_Integer:3] :=
  Module[{lhsExplicit, rhsExplicit},
    (* Check if indices are explicit *)
    lhsExplicit = IsExplicitIndices[lhs];
    rhsExplicit = IsExplicitIndices[rhs];
    
    Which[
      (* Case 1: Both have explicit indices *)
      lhsExplicit && rhsExplicit,
      Echo["Testing with explicit indices..."];
      TestOperatorEqualityExplicit[lhs, rhs],
      
      (* Case 2: At least one has implicit indices *)
      !lhsExplicit || !rhsExplicit,
      Echo["Testing with implicit indices (Einstein notation)..."];
      Echo[StringJoin[{"Expanding for dimensions 1 to ", ToString[maxDim]}]];
      TestOperatorEqualityImplicit[lhs, rhs, maxDim],
      
      (* Default *)
      True,
      Echo["Unable to determine index type"];
      False
    ]
  ];


(* ::Section:: *)
(*Advanced Testing Utilities*)


(* ::Subsection:: *)
(*Test on Fock States*)


(* Generate all occupation number matrices with total excitations <= Lambda *)
GenerateOccupationMatrices[dim_Integer, Lambda_Integer] :=
  Module[{n, tuples},
    n = dim^2;
    (* Generate all tuples with sum <= Lambda *)
    tuples = Select[Tuples[Range[0, Lambda], n], Total[#] <= Lambda &];
    (* Reshape into matrices *)
    Partition[#, dim]& /@ tuples
  ];

(* Test operator equality by evaluating on Fock states *)
TestOnFockStates[lhs_, rhs_, dim_Integer, Lambda_Integer] :=
  Module[{states, lhsExpanded, rhsExpanded, tests},
    (* Expand if needed *)
    lhsExpanded = If[IsExplicitIndices[lhs], 
      lhs, 
      ExpandImplicitIndices[lhs, dim]
    ];
    rhsExpanded = If[IsExplicitIndices[rhs], 
      rhs, 
      ExpandImplicitIndices[rhs, dim]
    ];
    
    (* Generate states *)
    states = Ket /@ GenerateOccupationMatrices[dim, Lambda];
    
    (* Test on each state *)
    tests = Table[
      Simplify[lhsExpanded ** state - rhsExpanded ** state] === 0,
      {state, states}
    ];
    
    If[AllTrue[tests, #&],
      Echo[StringJoin[{"All tests passed on Fock space (dim=", ToString[dim], ", Lambda=", ToString[Lambda], ")"}]];
      True,
      Echo["Some tests failed on Fock space"];
      False
    ]
  ];


(* ::Subsection:: *)
(*Batch Testing*)


TestMultipleEqualities[testList_List, maxDim_Integer:3] :=
  Module[{results, i, test, lhs, rhs},
    results = Table[
      test = testList[[i]];
      lhs = test[[1]];
      rhs = test[[2]];
      
      Echo["\n========================================"];
      Echo[StringJoin[{"Test ", ToString[i], "/", ToString@Length[testList]}]];
      Echo[StringJoin[{"LHS: ", ToString[lhs]}]];
      Echo[StringJoin[{"RHS: ", ToString[rhs]}]];
      Echo["========================================"];
      
      {i, TestOperatorEquality[lhs, rhs, maxDim]},
      {i, 1, Length[testList]}
    ];
    
    Echo["\n========================================"];
    Echo[StringJoin[{"Summary: ", ToString@Count[results[[All, 2]], True], "/", ToString@Length[testList], " tests passed"}]];
    Echo["========================================"];
    
    results
  ];


End[];
EndPackage[];
