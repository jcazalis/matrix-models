(* ::Package:: *)

(* ::Title:: *)
(*Ladder Operator Algebra*)


(* ::Author:: *)
(*Author: Jean Cazalis*)


(* ::Affiliation:: *)
(*Affiliation: DESY, CQTA*)


(* ::Abstract:: *)
(*This  package  implements  a  complete  non - commutative  algebraic  calculus  for  ladder  operators  indexed  by  two  numbers  (matricial  degree  of  freedom) . It  provides :*)
(*1)  Rules  for  non - commutative  multiplication  with  ladder  operators  (a  and  ad)*)
(*2)  Fock  state  machinery  including  ket  states, bra  states, and  their  adjoints*)
(*3)  Operator  actions  on  both  ket  and  bra  states*)
(*4)  Physical  operators : number  operators, quadrature  operators, matrix operators (X, P, A, SuperDagger[A])*)


(* ::Chapter:: *)
(*Package Public Interface*)


BeginPackage["LadderAlgebra`"];

(* Operators *)
a::usage = "a[i,j] represents the annihilation operator with lower index i and upper index j.";
ad::usage = "ad[i,j] represents the creation operator with lower index j and upper index i.";

(* Matrix dimension parameter *)
d::usage = "d represents the dimension of matrices in the matrix model.";

(* States *)
Ket::usage = "Ket[occ] represents a ket state |occ\:27e9 where occ is a matrix of occupation numbers.";
Bra::usage = "Bra[occ] represents a bra state \:27e8occ| where occ is a matrix of occupation numbers.";
VacuumKet::usage = "VacuumKet[n,m] creates the vacuum ket state |0\:27e9 with n\[Times]m matrix of zeros.";
VacuumBra::usage = "VacuumBra[n,m] creates the vacuum bra state \:27e80| with n\[Times]m matrix of zeros.";
Adjoint::usage = "Adjoint[state] returns the adjoint of a state (ket \[LeftRightArrow] bra).";

(* Commutation relation *)
CommutationRelation::usage = "CommutationRelation represents the commutation rule [a[i,j], ad[k,l]] = \[Delta]_i^l \[Delta]_k^j.";
AntiNormalOrder::usage = "AntiNormalOrder[expr] applies commutation relations repeatedly to achieve antinormal ordering (creation operators on the right).";
SimplifyNC::usage = "SimplifyNC[expr] simplifies an expression while applying commutation relations.";
\[Delta]::usage = "\[Delta][i,j] represents the Kronecker delta function. Evaluates to 1 if i=j (for integers), 0 otherwise. For symbolic indices, remains unevaluated.";

(* Physical operators *)
NumberOperator::usage = "NumberOperator[i,j] represents the number operator n_i^j = ad[j,i] ** a[i,j] for mode [i,j].";
TotalNumberOperator::usage = "TotalNumberOperator[n,m] represents the total number operator N = \[CapitalSigma]_{i,j} n_i^j for an n\[Times]m system.";
HarmonicOscillator::usage = "HarmonicOscillator[i,j] represents the number operator H_i^j = m(n_i^j + 1/2) for mode [i,j].";
TotalHarmonicOscillator::usage = "TotalHarmonicOscillator[n,m] represents the total number operator H = \[CapitalSigma]_{i,j} H_i^j for an n\[Times]m system.";
QuadratureX::usage = "QuadratureX[i,j,m] represents the position quadrature operator X_{ij} = (A_{ij} + A_{ij}\[Dagger])/\[Sqrt](2m).";
QuadratureP::usage = "QuadratureP[i,j,m] represents the momentum quadrature operator P_{ij} = i\[Sqrt](m/2)(A_{ij}\[Dagger] - A_{ij}).";
MatrixA::usage = "MatrixA[n,m] creates an n\[Times]m matrix of annihiliation operators.";
MatrixAd::usage = "MatrixAd[n,m] creates an n\[Times]m matrix of creation operators.";
MatrixX::usage = "MatrixX[n,m,mass] creates an n\[Times]m matrix of position quadrature operators.";
MatrixP::usage = "MatrixP[n,m,mass] creates an n\[Times]m matrix of momentum quadrature operators.";
NCMatrixMultiply::usage = "NCMatrixMultiply[A, B] performs non-commutative matrix multiplication where elements are multiplied using **.";
NCMatrixPower::usage = "NCMatrixPower[M, n] computes the n-th power of matrix M using non-commutative multiplication.";
NCTr::usage = "NCTr[M] computes the trace of a matrix of operators.";
ExpectationValue::usage = "ExpectationValue[operator, state] computes \[LeftAngleBracket]state|operator|state\[RightAngleBracket].";
MatrixElement::usage = "MatrixElement[bra, operator, ket] computes \[LeftAngleBracket]bra|operator|ket\[RightAngleBracket].";
trSigma::usage = "trSigma[\[Sigma], X1, X2, ..., Xn] computes the trace Tr(\[Sigma] X1\[TensorProduct]X2\[TensorProduct]...\[TensorProduct]Xn) where \[Sigma] is a permutation in Cycles form and Xi are square matrices with implicit indices.";
trSigmaExplicit::usage = "Not implemented yet.";
PermuteTensorIndices::usage = "PermuteTensorIndices[\[Sigma], indexList] applies permutation \[Sigma] to a list of tensor indices.";


(* ::Chapter:: *)
(*Package Implementation*)


Begin["`Private`"];


(* ::Section:: *)
(*Part 1: Non-Commutative Ladder Operator Algebra*)


(* ::Subsection:: *)
(*Initialize and Format Operators*)


(* Unprotect symbols for definition *)
Unprotect[NonCommutativeMultiply, Ket, Bra];
ClearAll[NonCommutativeMultiply, Ket, Bra, a, ad];

a::usage = "a[i,j] represents the annihilation operator with lower index i and upper index j.";
ad::usage = "ad[i,j] represents the creation operator with lower index j and upper index i.";

(* Set printing options to match LaTeX notation *)
Format[LadderAlgebra`ad[l_, k_], TraditionalForm] := Subsuperscript[SuperDagger[a], l, k]
Format[LadderAlgebra`a[l_, k_], TraditionalForm] := Subsuperscript[a, l, k]


(* ::Subsection:: *)
(*Non-Commutative Multiplication Rules*)


(* Rule for a single argument: ** |State\:27e9 -> |State\:27e9 *)
NonCommutativeMultiply[a_] := a;

(* Rule for distributivity: a ** (b + c) -> a ** b + a ** c *)
NonCommutativeMultiply[f___, g_Plus, h___] := 
  Plus @@ (NonCommutativeMultiply[f, #, h] & /@ List @@ g);

(* Rule to pull out raw scalar (c-number) coefficients *)
NonCommutativeMultiply[f___, c_?NumericQ, h__] /; (h =!= Sequence[]) := 
  c * NonCommutativeMultiply[f, h];

(* Rule to pull out scalar from a Times expression: a ** (c * b) -> c * (a ** b) *)
NonCommutativeMultiply[f___, c_?NumericQ * g_, h___] := 
  c * NonCommutativeMultiply[f, g, h];

(* Commutation rules: creation operators commute with each other *)
(* ad[i1,j1] ** ad[i2,j2] = ad[i2,j2] ** ad[i1,j1] if not already ordered *)
NonCommutativeMultiply[left___, ad[i1_, j1_], ad[i2_, j2_], right___] /; 
  !OrderedQ[{{i1, j1}, {i2, j2}}] := 
    NonCommutativeMultiply[left, ad[i2, j2], ad[i1, j1], right];

(* Commutation rules: annihilation operators commute with each other *)
(* a[i1,j1] ** a[i2,j2] = a[i2,j2] ** a[i1,j1] if not already ordered *)
NonCommutativeMultiply[left___, a[i1_, j1_], a[i2_, j2_], right___] /; 
  !OrderedQ[{{i1, j1}, {i2, j2}}] := 
    NonCommutativeMultiply[left, a[i2, j2], a[i1, j1], right];

(* Rule to handle a numeric constant acting on a state: c ** |state\:27e9 -> c |state\:27e9 *)
NonCommutativeMultiply[c_?NumericQ, state_Ket] := c * state;
NonCommutativeMultiply[c_?NumericQ, state_Bra] := c * state;

NonCommutativeMultiply[f___, c_?NumericQ, state_Ket] := 
  NonCommutativeMultiply[f, c * state];

NonCommutativeMultiply[f___, c_?NumericQ, state_Bra] := 
  NonCommutativeMultiply[f, c * state];

(* Rule to annihilate any expression multiplied by zero *)
NonCommutativeMultiply[___, 0, ___] := 0;

(* Empty multiplication is one *)
NonCommutativeMultiply[] = 1;

(* Set attributes for NonCommutativeMultiply:
   Flat: a ** b ** c is treated as NonCommutativeMultiply[a, b, c]
   OneIdentity: 1 ** a -> a and a ** 1 -> a
*)
SetAttributes[NonCommutativeMultiply, {Flat, OneIdentity}];


(* ::Subsection:: *)
(*Commutation Relations*)


(* Define the commutation relation [a, ad] = \[Delta] *)
(* This rule will replace ad ** a with a ** ad - \[Delta] *)
CommutationRelation = {
  NonCommutativeMultiply[left___, ad[l_, k_], a[n_, m_], right___] :> 
    NonCommutativeMultiply[left, a[n, m], ad[l, k], right] - 
    NonCommutativeMultiply[left, right] * \[Delta][l, m] * \[Delta][n, k]
};

(* AntiNormal ordering: move all creation operators (ad) to the right *)
AntiNormalOrder[expr_] := FixedPoint[# //. CommutationRelation &, expr];

AntiNormalOrder::usage = 
  "AntiNormalOrder[expr] applies commutation relations repeatedly to move all \
creation operators to the right of annihilation operators (antinormal ordering).";

(* Simplify with commutation relations *)
SimplifyNC[expr_] := Simplify[AntiNormalOrder[expr]];

SimplifyNC::usage = 
  "SimplifyNC[expr] simplifies an expression while applying commutation relations \
to achieve antinormal ordering.";


(* ::Subsection:: *)
(*Custom Delta Function for Symbolic Indices*)


(* Define custom delta function that handles symbolic indices properly *)
(* We use \[Delta] instead of trying to override KroneckerDelta *)
Unprotect[\[Delta]];
ClearAll[\[Delta]];

(* Only evaluate when both arguments are integers *)
\[Delta][i_Integer, j_Integer] := If[i === j, 1, 0];

(* For symbolic arguments, no rule is defined, so it stays unevaluated *)

(* Formatting for \[Delta] in TraditionalForm *)
Format[\[Delta][i_, j_], TraditionalForm] := Subsuperscript[\[Delta], i, j];

Protect[\[Delta]];


(* ::Section:: *)
(*Part 2: Fock State Machinery*)


(* ::Subsection:: *)
(*State Definitions and Formatting*)


(* Format for ket states *)
Format[Ket[occ_?MatrixQ]] := 
  "\[LeftBracketingBar]" <> ToString[occ, TraditionalForm] <> "\[RightAngleBracket]";

(* Format for bra states *)
Format[Bra[occ_?MatrixQ]] := 
  "\[LeftAngleBracket]" <> ToString[occ, TraditionalForm] <> "\[RightBracketingBar]";

(* Define vacuum ket state *)
VacuumKet[n_Integer, m_Integer] := Ket[ConstantArray[0, {n, m}]];

(* Define vacuum bra state *)
VacuumBra[n_Integer, m_Integer] := Bra[ConstantArray[0, {n, m}]];


(* ::Subsection:: *)
(*Adjoint Operations*)


(* Adjoint of a ket state gives a bra state *)
Adjoint[Ket[occ_?MatrixQ]] := Bra[occ];

(* Adjoint of a bra state gives a ket state *)
Adjoint[Bra[occ_?MatrixQ]] := Ket[occ];

(* Adjoint is linear *)
Adjoint[c_?NumericQ * state_] := Conjugate[c] * Adjoint[state];
Adjoint[sum_Plus] := Adjoint /@ sum;

(* Adjoint distributes over Times with symbolic scalar factors (e.g. d * a[1,2]) *)
Adjoint[c_ * expr_] /; FreeQ[c, a | ad | Ket | Bra | NonCommutativeMultiply] :=
  Conjugate[c] * Adjoint[expr];

(* Adjoint of operators *)
Adjoint[a[i_, j_]] := ad[j, i];
Adjoint[ad[i_, j_]] := a[j, i];

(* Adjoint reverses order in non-commutative products *)
Adjoint[expr_NonCommutativeMultiply] := 
  Module[{args = List @@ expr},
    NonCommutativeMultiply @@ Reverse[Adjoint /@ args]
  ];

(* Fallback: purely scalar expressions (symbolic or numeric) *)
Adjoint[c_] /; FreeQ[c, a | ad | Ket | Bra | NonCommutativeMultiply] := Conjugate[c];


(* ::Subsection:: *)
(*Operator Actions on Ket States*)


(* Action of Creation Operator ad[k2,k1] on ket *)
(* Beware of the effect of the conventions here: up and down indices are inverted *)
ad[k2_, k1_] ** Ket[occ_] :=
  Module[{newOcc = occ, n},
    n = occ[[k1, k2]]; (* Get current occupation n_k1,k2 *)
    newOcc[[k1, k2]] = n + 1; (* Increment it *)
    Sqrt[n + 1] * Ket[newOcc] (* Return Sqrt[n+1]*|n+1\:27e9 *)
  ];

(* Action of Annihilation Operator a[k1,k2] on ket *)
a[k1_, k2_] ** Ket[occ_] :=
  Module[{newOcc = occ, n},
    n = occ[[k1, k2]]; (* Get current occupation n_k1,k2 *)
    If[n == 0,
      0, (* a|0\:27e9 = 0 *)
      newOcc[[k1, k2]] = n - 1; (* Decrement it *)
      Sqrt[n] * Ket[newOcc] (* Return Sqrt[n]* |n-1\:27e9 *)
    ]
  ];


(* ::Subsection:: *)
(*Operator Actions on Bra States*)


(* Action of Creation Operator ad on bra: \:27e8occ| ad = \:27e8occ-1| Sqrt[n] *)
(* This is derived from (ad|occ\:27e9)\[Dagger] = \:27e8occ|ad\[Dagger] = \:27e8occ|a *)
Bra[occ_] ** ad[k2_, k1_] :=
  Module[{newOcc = occ, n},
    n = occ[[k1, k2]]; (* Get current occupation n_k1,k2 *)
    If[n == 0,
      0, (* \:27e80|ad = 0 *)
      newOcc[[k1, k2]] = n - 1; (* Decrement it *)
      Sqrt[n] * Bra[newOcc] (* Return Sqrt[n]* \:27e8n-1| *)
    ]
  ];

(* Action of Annihilation Operator a on bra: \:27e8occ| a = \:27e8occ+1| Sqrt[n+1] *)
Bra[occ_] ** a[k1_, k2_] :=
  Module[{newOcc = occ, n},
    n = occ[[k1, k2]]; (* Get current occupation n_k1,k2 *)
    newOcc[[k1, k2]] = n + 1; (* Increment it *)
    Sqrt[n + 1] * Bra[newOcc] (* Return Sqrt[n+1]* \:27e8n+1| *)
  ];


(* ::Subsection:: *)
(*Rules for States in NonCommutativeMultiply*)


(* Rule for an operator acting on a ket state that is already scaled *)
NonCommutativeMultiply[f___, op : ((a | ad)[_, _]), (c_?NumericQ * h_Ket)] := 
  c * NonCommutativeMultiply[f, op, h];

(* Rule for a bra state acting on an operator that is already scaled *)
NonCommutativeMultiply[f___, (c_?NumericQ * h_Bra), op : ((a | ad)[_, _]), rest___] := 
  c * NonCommutativeMultiply[f, h, op, rest];

(* The main recursive rule for operators on ket: op1 ** op2 ** |Ket\:27e9 *)
NonCommutativeMultiply[f___, op : ((a | ad)[_, _]), h_Ket] := 
  NonCommutativeMultiply[f, (op ** h)];

(* The main recursive rule for bra on operators: \:27e8State| ** op1 ** op2 *)
NonCommutativeMultiply[f___, h_Bra, op : ((a | ad)[_, _]), rest___] := 
  NonCommutativeMultiply[f, (h ** op), rest];

(* Inner product: \:27e8m|n\:27e9 = \[Delta]_mn *)
NonCommutativeMultiply[f___, Bra[occ1_?MatrixQ], Ket[occ2_?MatrixQ], rest___] := 
  If[occ1 === occ2,
    NonCommutativeMultiply[f, rest],
    0
  ];


(* ::Section:: *)
(*Part 3: Physical Operators*)


(* ::Subsection:: *)
(*Number Operators*)


(* Number operator for mode [i,j]: n_i^j = ad[j,i] ** a[i,j] *)
(* Handle both explicit (integer) and implicit (symbolic) indices *)
NumberOperator[i_, j_] := ad[j, i] ** a[i, j];

(* Formatting for number operator *)
Format[HoldPattern[NumberOperator[i_, j_]], TraditionalForm] := Subsuperscript[n, i, j];

(* Total number operator: N = \[CapitalSigma]_{i,j} n_i^j *)
TotalNumberOperator[n_Integer, m_Integer] := 
  Sum[NumberOperator[i, j], {i, 1, n}, {j, 1, m}];

(* Formatting for total number operator *)
Format[TotalNumberOperator[n_, m_], TraditionalForm] := 
  OverscriptBox[N, Row[{n, "\[Times]", m}]];


(* ::Subsection:: *)
(*Quantum Harmonic Oscillators*)


(* Quantum harmonic oscillators for mode [i,j]: H_i^j = m(n_i^j + 1/2)*)
(* Handle both explicit (integer) and implicit (symbolic) indices *)
HarmonicOscillator[i_, j_, mass_] := mass*(NumberOperator[i, j] + 1/2);

(* Formatting for quantum harmonic oscillator operator *)
Format[HoldPattern[HarmonicOscillator[i_, j_, m_]], TraditionalForm] := Subsuperscript[H, i, j];

(* Total number operator: H = \[CapitalSigma]_{i,j} H_i^j *)
TotalHarmonicOscillator[n_Integer, m_Integer, mass_] := 
  Sum[HarmonicOscillator[i, j, mass], {i, 1, n}, {j, 1, m}];

(* Formatting for total number operator *)
Format[TotalHarmonicOscillator[n_, m_], TraditionalForm] := 
  OverscriptBox[H, Row[{n, "\[Times]", m}]];


(* ::Subsection:: *)
(*Quadrature Operators*)


(* Position quadrature operator X_i^j = (A_{ij} + A_{ij}\[Dagger])/\[Sqrt](2m) *)
(* Handle both explicit (integer) and implicit (symbolic) indices *)
QuadratureX[i_, j_, mass_] := (a[i, j] + ad[j, i]) / Sqrt[2 * mass];

(* Momentum quadrature operator P_i^j = i\[Sqrt](m/2)(A_{ij}\[Dagger] - A_{ij}) *)
(* Handle both explicit (integer) and implicit (symbolic) indices *)
QuadratureP[i_, j_, mass_] := I * Sqrt[mass / 2] * (ad[j, i] - a[i, j]);

(* Formatting for quadrature operators *)
Format[HoldPattern[QuadratureX[i_, j_, m_]], TraditionalForm] := Subsuperscript[X, i, j];
Format[HoldPattern[QuadratureP[i_, j_, m_]], TraditionalForm] := Subsuperscript[P, i, j];


(* ::Subsection:: *)
(*Matrix Operators*)


(* Full matrix A operator (n\[Times]m system) *)
MatrixA[n_Integer, m_Integer] := 
  Table[a[i, j], {i, 1, n}, {j, 1, m}];

(* Full matrix A operator (n\[Times]m system) *)
MatrixAd[n_Integer, m_Integer] := 
  Table[ad[i, j], {i, 1, n}, {j, 1, m}];

(* Full matrix X operator (n\[Times]m system) *)
MatrixX[n_Integer, m_Integer, mass_] := 
  (MatrixA[n, m] + MatrixAd[n, m]) / Sqrt[2 * mass];

(* Full matrix P operator (n\[Times]m system) *)
MatrixP[n_Integer, m_Integer, mass_] := 
  I * Sqrt[mass / 2] * (MatrixAd[n, m] - MatrixA[n, m]);


(* ::Subsection:: *)
(*Matrix Operations for Operators*)


(* Non-commutative matrix multiplication for matrices of operators *)
(* Uses ** for element multiplication instead of regular * *)
NCMatrixMultiply[A_?MatrixQ, B_?MatrixQ] := 
  Module[{rowsA, colsA, rowsB, colsB},
    {rowsA, colsA} = Dimensions[A];
    {rowsB, colsB} = Dimensions[B];
    
    If[colsA =!= rowsB,
      Message[NCMatrixMultiply::incompatible, Dimensions[A], Dimensions[B]];
      Return[$Failed]
    ];
    
    Table[
      Sum[A[[i, k]] ** B[[k, j]], {k, 1, colsA}],
      {i, 1, rowsA}, {j, 1, colsB}
    ]
  ];

NCMatrixMultiply::incompatible = 
  "Incompatible matrix dimensions: `1` and `2` cannot be multiplied.";

NCMatrixMultiply::usage = 
  "NCMatrixMultiply[A, B] performs non-commutative matrix multiplication where \
elements are multiplied using NonCommutativeMultiply (**).";

(* Non-commutative matrix power *)
NCMatrixPower[M_?MatrixQ, n_Integer] := 
  Which[
    n == 0, IdentityMatrix[Length[M]],
    n == 1, M,
    n > 1, Fold[NCMatrixMultiply, M, Table[M, {n - 1}]],
    n < 0, Message[NCMatrixPower::negexp]; $Failed
  ];

NCMatrixPower::negexp = "Negative exponents not supported for operator matrices.";

NCMatrixPower::usage = 
  "NCMatrixPower[M, n] computes the n-th power of matrix M using non-commutative multiplication.";

(* Trace of a matrix of operators *)
NCTr[M_?MatrixQ] := Sum[M[[i, i]], {i, 1, Min[Dimensions[M]]}];

NCTr::usage = "NCTr[M] computes the trace of a matrix of operators.";


(* ::Subsection:: *)
(*Expectation Values*)


(* Helper function to compute expectation value \[LeftAngleBracket]\[Psi]|O|\[Psi]\[RightAngleBracket] *)
ExpectationValue[operator_, state_Ket] := 
  Adjoint[state] ** operator ** state;

(* Matrix element \[LeftAngleBracket]\[Psi]|O|\[Phi]\[RightAngleBracket] *)
MatrixElement[bra_Bra, operator_, ket_Ket] := 
  bra ** operator ** ket;


(* ::Subsection:: *)
(*Permuted Tensor Product Traces*)


(* Helper function: Apply permutation to list of indices *)
PermuteTensorIndices[perm_Cycles, indexList_List] := 
  Module[{n = Length[indexList], permutedIndices},
    (* Convert Cycles to permutation list *)
    permutedIndices = PermutationList[perm, n];
    (* Apply permutation *)
    indexList[[permutedIndices]]
  ];

(* Implicit index version: trSigma[\[Sigma], X1, X2, ..., Xn] *)
(* Computes Tr(\[Sigma] X_1 \[TensorProduct] X_2 \[TensorProduct] ... \[TensorProduct] X_n) with implicit indices *)

(* Special case: no matrices returns 0 *)
trSigma[perm_Cycles] := 1;

trSigma[perm_Cycles, matrices__] := 
  Module[{n, indexList, permutedIndices, terms, result},
    (* Number of matrices in tensor product *)
    n = Length[{matrices}];
    
    (* Create symbolic index list {k1, k2, ..., kn} *)
    indexList = Table[Symbol["k" <> ToString[i]], {i, 1, n}];
    
    (* Apply permutation to indices *)
    permutedIndices = PermuteTensorIndices[perm, indexList];
    
    (* Build the trace expression: sum over indices *)
    (* Tr(\[Sigma] X1\[TensorProduct]X2\[TensorProduct]...\[TensorProduct]Xn) = \[CapitalSigma]_{k1,...,kn} X1_{k1,\[Sigma](k)_1} X2_{k2,\[Sigma](k)_2} ... Xn_{kn,\[Sigma](k)_n} *)
    
    terms = MapThread[#1[#2,#3]&, {{matrices}, indexList, permutedIndices}];
    result = Distribute[Apply[NonCommutativeMultiply,terms]];
    result
  ];

(* Explicit index version: trSigmaExplicit[\[Sigma], n, X1, X2, ..., Xn] *)
(* Computes Tr(\[Sigma] X_1 \[TensorProduct] X_2 \[TensorProduct] ... \[TensorProduct] X_n) with explicit summation *)
trSigmaExplicit[perm_Cycles, dim_Integer, matrices__] := Null;

trSigma::usage = 
  "trSigma[\[Sigma], X1, X2, ..., Xn] computes the trace Tr(\[Sigma] X1\[TensorProduct]X2\[TensorProduct]...\[TensorProduct]Xn) \
where \[Sigma] is a permutation in Cycles form and Xi are abstract square matrices (linear combination of a and ad). \
The trace is computed with implicit indices summed over matrix dimensions.";

trSigmaExplicit::usage = 
  "trSigmaExplicit[\[Sigma], dim, X1, X2, ..., Xn] computes the trace Tr(\[Sigma] X1\[TensorProduct]X2\[TensorProduct]...\[TensorProduct]Xn) \
where \[Sigma] is a permutation in Cycles form, dim is the matrix dimension, \
and Xi are square matrices. The trace is computed with explicit summation from 1 to dim.";

(* Formatting for trSigma operators *)
Format[HoldPattern[trSigma[perm_Cycles, matrices__]], TraditionalForm] := 
  SubscriptBox["Tr", perm][Subscript["X", 1] \[TensorProduct] Subscript["X", 2] \[TensorProduct] "..." \[TensorProduct] Subscript["X", Length[{matrices}]]];

Format[HoldPattern[trSigmaExplicit[perm_Cycles, dim_, matrices__]], TraditionalForm] := 
  SubscriptBox["Tr", perm][Subscript["X", 1] \[TensorProduct] Subscript["X", 2] \[TensorProduct] "..." \[TensorProduct] Subscript["X", Length[{matrices}]]];


(* ::Subsection:: *)
(*Protect Symbols*)


(* Protect the symbols *)
Protect[NonCommutativeMultiply, Ket, Bra, NumberOperator, TotalNumberOperator, 
  QuadratureX, QuadratureP, MatrixA, MatrixAd, MatrixX, MatrixP, HarmonicOscillator, TotalHarmonicOscillator, 
  NCMatrixMultiply, NCMatrixPower, NCTr, AntiNormalOrder, SimplifyNC, ExpectationValue, MatrixElement,
  trSigma, trSigmaExplicit, PermuteTensorIndices, Adjoint];


End[];
EndPackage[];
