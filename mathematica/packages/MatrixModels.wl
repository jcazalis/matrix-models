(* ::Package:: *)

(* ::Title:: *)
(*Matrix Models Package*)


(* ::Author:: *)
(*Author: Jean Cazalis*)


(* ::Affiliation:: *)
(*Affiliation: DESY, CQTA*)


(* ::Abstract:: *)
(*This package implements tools for working with U(N) matrix models.*)
(*It builds on LadderAlgebra` and provides:*)
(*1) Decomposition of cyclic products Tr(\[Sigma] X\[TensorProduct]P\[TensorProduct]...\[TensorProduct]X) into anti-normal ordered generalized traces*)
(*2) Tensor contraction machinery for overlap calculations*)
(*3) Permuted tensor product traces with symmetry considerations*)


(* ::Chapter:: *)
(*Package Public Interface*)


BeginPackage["MatrixModels`", {"LadderAlgebra`", "OperatorTesting`"}];

(* Core decomposition functions *)
MonomialTrace::usage = "MonomialTrace[ops, perm, mass] creates the symbolic representation of Tr_perm(M1\[CircleTimes]M2\[CircleTimes]...\[CircleTimes]Mk) where ops is a string of X/P operators (e.g. \"XXXX\", \"XPXP\") and perm is a 1-indexed permutation list. Default mass is 1. Example: MonomialTrace[\"XPXP\", {2,3,4,1}, 1/2].";
CyclicProduct::usage = "CyclicProduct[k, mass] convenience wrapper for MonomialTrace with all-X operators and cyclic permutation. Equivalent to MonomialTrace[StringJoin[Table[\"X\",k]], PermutationList[Cycles[{Range[k]}],k], mass].";
EvenMonomialQ::usage = "EvenMonomialQ[ops] returns True if the operator string ops has even length, i.e. the monomial has an even number of operators.";
HermitianMonomialQ::usage = "HermitianMonomialQ[ops, perm] returns True if the monomial trace Tr_perm(M(ops)) is a Hermitian operator. Default mass is 1.";
DecomposeTrace::usage = "DecomposeTrace[ops, perm, mass, opts] decomposes Tr_perm(M(X,P)) into anti-normal ordered generalized traces. ops is a string of X/P operators, perm is a 1-indexed permutation list. Default mass is 1. Options: Verbose->False, PrintTable->False, Simplify->True.";
ReconstructTrace::usage = "ReconstructTrace[decomposition] reconstructs the full expression from a decomposition (type -> {perm -> coeff}).";

(* Utility functions for decomposition *)
ContractIndices::usage = "ContractIndices[expr] contracts Kronecker deltas in an expression, replacing equivalent indices.";
IdentifyTerms::usage = "IdentifyTerms[expr] identifies terms by their operator type, index permutation, and coefficient.";
CollectTermsByType::usage = "CollectTermsByType[identifiedTerms] aggregates terms with the same type and permutation.";
SimplifyCollectedTerms::usage = "SimplifyCollectedTerms[collectedTerms] simplifies the collected terms by grouping permutations that are equivalent under the symmetry group S_nA \[Times] S_nAd for each operator type.";

(* Overlap calculation functions *)
DeclareYoungProjector::usage = "DeclareYoungProjector[n, isLeft] creates a Young projector symbol with n operators. If isLeft=True, uses annihilation operators (left state). If isLeft=False, uses creation operators (right state).";
ApplyWickTheorem::usage = "ApplyWickTheorem[expr] applies Wick's theorem to contract all operators, replacing them with appropriate Kronecker deltas.";
RelabelIndices::usage = "RelabelIndices[expr] relabels all indices with k1, k2, etc. using canonical ordering based on down indices.";
ConvertToTensorContractions::usage = "ConvertToTensorContractions[expr] converts Young projector products into abstract tensor contractions.";
CanonicalizeTensorExpression::usage = "CanonicalizeTensorExpression[expr, nLeft, nRight] applies TensorReduce to canonicalize tensor contraction expressions with left and right Young projectors of sizes nLeft and nRight.";
ComputeOverlap::usage = "ComputeOverlap[nLeft, type, sigma, opts] computes the overlap matrix element \[LeftAngleBracket]nLeft|Tr_\[Sigma](type)|nRight\[RightAngleBracket] where nRight is determined by excitation balance. Options: Verbose->False, Efficient->True.";

(* Matrix element calculation functions *)
PermutationTensorProduct::usage = "PermutationTensorProduct[sigma1, sigma2] computes the tensor product of two permutations.";
GetRepresentationIndex::usage = "GetRepresentationIndex[partition, n] finds the index of the representation corresponding to partition in S_n.";
ContractPermutationTensor::usage = "ContractPermutationTensor[sigma, pairs, n] computes the sum of contractions for each pairing pattern in pairs.";
GetPairsAndCoefficientsAssociation::usage = "GetPairsAndCoefficientsAssociation[expr] extracts all contraction patterns and their coefficients from a canonicalized tensor expression.";
ComputeContractions::usage = "ComputeContractions[R, S, pairs] computes the sum of tensor contractions weighted by character theory coefficients for partitions R (left) and S (right).";
ComputeSingleOverlap::usage = "ComputeSingleOverlap[R, S, type, sigma, opts] computes a single overlap matrix element \[LeftAngleBracket]R|Tr_\[Sigma](type)|S\[RightAngleBracket] where R and S are partitions. Options: Verbose->False.";
ComputeMatrixElementInteraction::usage = "ComputeMatrixElementInteraction[R, S, ops, perm, opts] computes the matrix element \[LeftAngleBracket]R|Tr_perm(M(ops))|S\[RightAngleBracket] by decomposing the monomial and summing over all overlap contributions. ops is a string of X/P operators, perm is a 1-indexed permutation list. Options: Verbose->False, Efficient->True, UseWovenContractions->True.";

(* Hamiltonian matrix construction *)
ComputeHInt::usage = "ComputeHInt[ops, perm, maxLambda, opts] computes the matrix of Tr_perm(M(ops)) in the Young basis up to maxLambda excitations. ops is a string of X/P operators, perm is a 1-indexed permutation list. Options: Verbose->False, PrecomputedContractions->None, GroundStateOnly->False. If PrecomputedContractions is an Association with keys {pair, R, S} -> polynomial in d, those values are reused. If GroundStateOnly->True, only even-excitation partitions are included.";
ComputeHfree::usage = "ComputeHfree[maxLambda, opts] computes the free Hamiltonian in the Young basis up to maxLambda excitations. Options: GroundStateOnly->False. Returns a diagonal matrix with entries d^2/2 + Total[R]. If GroundStateOnly->True, only even-excitation partitions are included.";
PartitionList::usage = "PartitionList[maxLambda] returns the list of all integer partitions from 0 to maxLambda, in the order used by ComputeHInt and ComputeHfree.";

(* Woven contraction functions *)
ComputeContributingWovenContractions::usage = "ComputeContributingWovenContractions[nL, nR, collectedTerms] computes all woven contractions contributing to \[LeftAngleBracket]R|Tr_perm(M(ops))|S\[RightAngleBracket] where R has excitation level nL and S has excitation level nR. collectedTerms is the output of DecomposeTrace. Returns an Association mapping contraction pair patterns to polynomial coefficients in d.";
ComputeAllWovenContractions::usage = "ComputeAllWovenContractions[ops, perm, Lambda, mass, opts] computes all contributing woven contractions for Tr_perm(M(ops)) for all excitation levels nL, nR from 0 to Lambda. ops is a string of X/P operators, perm is a 1-indexed permutation list. Options: Verbose->False. Returns a nested Association <|{nL, nR} -> wovenContractions, ...|>.";
ExportWovenContractions::usage = "ExportWovenContractions[ops, perm, mass, wovenData, outputDir] exports woven contraction data to JSON for Python consumption. ops is a string of X/P operators, perm is a 1-indexed permutation list. Skips export if a file with same ops, perm, mass and larger or equal Lambda already exists. Permutations are stored as 0-indexed arrays. Coefficients are polynomial coefficient lists in d as [re_num, re_den, im_num, im_den] rational 4-tuples. File saved as wc_op_{ops}_p{perm}_m{mass}_Lambda{Lambda}.json.";
ImportWovenContractions::usage = "ImportWovenContractions[ops, perm, Lambda, inputDir] imports woven contraction data from JSON back into the same Association format produced by ComputeAllWovenContractions. ops is a string of X/P operators, perm is a 1-indexed permutation list. Coefficients are reconstructed as polynomials in d.";

(* Bruteforce calculations *)
ComputeBruteForce::usage = "ComputeBruteForce[R, S, ops, perm, dim, mass] computes the overlap <R|tr_perm(M(ops))|S> for partitions R (left) and S (right) with underlying space dimension equal to dim. ops is a string of X/P operators, perm is a 1-indexed permutation list.";

(* Symbols *)
d::usage = "d represents the dimension of matrices in the matrix model (same as LadderAlgebra`d).";
P::usage = "P represents Young projector tensors used in overlap calculations.";

(* Note: Using \[Delta], a, ad, trSigma from LadderAlgebra` package *)


(* ::Chapter:: *)
(*Package Implementation*)


Begin["`Private`"];


(* ::Section::Closed:: *)
(*Initialization*)


(* Import symbols from LadderAlgebra *)
(* a, ad, NonCommutativeMultiply, \[Delta] are already available *)
$PackageRoot = FileNameJoin[{DirectoryName[$InputFileName], "..", ".."}];


(* ::Section:: *)
(*Trace Decomposition*)


(* ::Subsection:: *)
(*Monomial Trace Construction*)


(* Main implementation: general monomial trace Tr_perm(M1 \[CircleTimes] M2 \[CircleTimes] ... \[CircleTimes] Mk) *)
MonomialTrace[ops_String, perm_List, mass_:1] := 
  Module[{k, opFns, myX, myP, result},
    k = StringLength[ops];
    If[k == 0, Return[d]];
    
    (* Define quadrature operators *)
    myX[i_, j_] := (LadderAlgebra`a[i, j] + LadderAlgebra`ad[i, j]) / Sqrt[2 * mass];
    myP[i_, j_] := I * Sqrt[mass / 2] * (LadderAlgebra`ad[i, j] - LadderAlgebra`a[i, j]);
    
    (* Parse operator string into function list *)
    opFns = Characters[ops] /. {"X" -> myX, "P" -> myP};
    
    (* Compute trace using permuted tensor product from LadderAlgebra *)
    result = LadderAlgebra`trSigma[
      PermutationCycles[perm],
      Sequence @@ opFns
    ] // Simplify;
    
    result
  ];

(* Convenience wrapper: cyclic trace of X^k *)
CyclicProduct[k_Integer, mass_:1] := 
  MonomialTrace[
    StringJoin[Table["X", k]],
    PermutationList[Cycles[{Range[k]}], k],
    mass
  ];


(* Even/odd parity: simply count the number of operators *)
EvenMonomialQ[ops_String] := EvenQ[StringLength[ops]];


(* Private helper: setwise stabilizer of the X-positions in ops under S_K.
   = permutations that send X-positions to X-positions (and P to P)
   = S_nX \[Times] S_nP \:21aa S_K.  GroupSetwiseStabilizer handles all the group
   theory internally; no manual generator construction needed. *)
getOpsStabilizerGroup[ops_String] :=
  Module[{k, xPos},
    k    = StringLength[ops];
    xPos = Flatten[Position[Characters[ops], "X"]];
    GroupSetwiseStabilizer[SymmetricGroup[k], xPos]
  ];

(* Hermitianity check: combinatorial group-orbit test.
   T\[Dagger] = Tr_{\[Sigma]\:207b\.b9}(O_K \[CircleTimes] \[Ellipsis] \[CircleTimes] O_1).  The monomial is Hermitian iff
   \[Sigma]\:207b\.b9 lies in the conjugation orbit of \[Sigma] under G_ops = S_nX \[Times] S_nP \:21aa S_K.
   The `mass` argument is kept for API compatibility but is unused:
   both X and P are Hermitian (X\[Dagger] = X, P\[Dagger] = P) regardless of mass. *)
HermitianMonomialQ[ops_String, perm_List, mass_:1] :=
  Module[{k, adjPerm, G, conjAction, orbit},
    k       = StringLength[ops];
    adjPerm = PermutationList[
                InversePermutation[PermutationCycles[perm]], k];
                
    (* Necessary condition: the string must be a palindrome. *)
    If[StringReverse[ops] =!= ops, Return[False]];

    (* Stabilizer G_ops = S_nX \[Times] S_nP \:21aa S_K *)
    G = getOpsStabilizerGroup[ops];

    (* Conjugation action: \[Pi] acts on \[Sigma] by \[Pi]\[CenterDot]\[Sigma]\[CenterDot]\[Pi]\:207b\.b9.
       In Mathematica's left-to-right PermutationProduct convention:
         PermutationProduct[\[Pi]\:207b\.b9, \[Sigma], \[Pi]][x] = \[Pi](\[Sigma](\[Pi]\:207b\.b9(x))) = (\[Pi] \[Sigma] \[Pi]\:207b\.b9)(x) *)
    conjAction = Function[{sigma, pi},
      PermutationList[
        PermutationProduct[
          InversePermutation[PermutationCycles[pi]],
          PermutationCycles[sigma],
          PermutationCycles[pi]
        ],
        k
      ]
    ];

    (* \[Sigma] is Hermitian iff \[Sigma]\:207b\.b9 is in the conjugation orbit of \[Sigma] under G *)
    orbit = GroupOrbits[G, {perm}, conjAction][[1]];
    MemberQ[orbit, adjPerm]
  ];


(* ::Subsection::Closed:: *)
(*Anti-Normal Ordering*)


(* Use the commutation relations from LadderAlgebra *)
(* Apply anti-normal ordering using LadderAlgebra's AntiNormalOrder function *)
antiNormalOrder[expr_] := LadderAlgebra`AntiNormalOrder[expr];


(* ::Subsection::Closed:: *)
(*Index Contraction*)


(* Get all index symbols from an expression (k1, k2, i1, j1, m1, n1, etc.) *)
getIndices[expr_] := 
  Union[
    Cases[expr, s_Symbol /; StringMatchQ[SymbolName[s], "k" ~~ DigitCharacter..], Infinity],
    Cases[expr, s_Symbol /; StringMatchQ[SymbolName[s], "i" ~~ DigitCharacter..], Infinity],
    Cases[expr, s_Symbol /; StringMatchQ[SymbolName[s], "j" ~~ DigitCharacter..], Infinity],
    Cases[expr, s_Symbol /; StringMatchQ[SymbolName[s], "m" ~~ DigitCharacter..], Infinity],
    Cases[expr, s_Symbol /; StringMatchQ[SymbolName[s], "n" ~~ DigitCharacter..], Infinity]
  ];

(* Remove deltas involving indices in the given list *)
clearDeltas[expr_, indexList_] := 
  DeleteCases[
    expr,
    (LadderAlgebra`\[Delta][arg1_, arg2_] | Power[LadderAlgebra`\[Delta][arg1_, arg2_], _Integer]) /; 
      MemberQ[indexList, arg1 | arg2],
    Infinity
  ];

(* Process a single term: contract indices according to deltas *)
contractTerm[term_] := 
  Module[{deltas, nonDeltaPart, edges, components, rules, contractedTerm, 
          remainingIndices, remainingDeltas, finalTerm},
    
    (* Separate deltas from the rest *)
    deltas = Cases[term, LadderAlgebra`\[Delta][_, _], {0,Infinity}];
    nonDeltaPart = term //. LadderAlgebra`\[Delta][_, _] -> 1;
    
    (* If no deltas, return as is *)
    If[Length[deltas] == 0, Return[term]];
    
    (* Build graph of index equivalences *)
    edges = deltas /. LadderAlgebra`\[Delta] -> UndirectedEdge;
    
    (* Find connected components (groups of equivalent indices) *)
    components = ConnectedComponents[Graph[edges]];
    
    (* Create replacement rules: replace all indices in a component with the canonical (first) one *)
    rules = Flatten[
      Function[component, 
        Thread[Rest[Sort[component]] -> First[Sort[component]]]
      ] /@ components
    ];
    
    (* Apply rules to contract indices *)
    contractedTerm = term /. rules;
    
    (* Now handle remaining deltas \[Delta][ki, ki] -> d *)
    remainingIndices = getIndices[nonDeltaPart /. rules];
    remainingDeltas = Cases[contractedTerm, LadderAlgebra`\[Delta][_, _], {0,Infinity}];
    remainingDeltas = clearDeltas[remainingDeltas, remainingIndices];
    
    (* Count remaining deltas and multiply by d^count *)
    finalTerm = d^Length[remainingDeltas] * (contractedTerm //. LadderAlgebra`\[Delta][_, _] -> 1);
    
    finalTerm
  ];

ContractIndices[expr_] := 
  Module[{terms},
    terms = MonomialList[Expand[expr]];
    Total[contractTerm /@ terms]
  ];


(* ::Subsection::Closed:: *)
(*Term Identification*)


(* Identify a single term: extract type, permutation, and coefficient *)
identifyTerm[term_] := 
  Module[{operators, coefficient, type, downIndices, upIndices, sigma},
    
    (* Separate operators from coefficient *)
    operators = Cases[term, LadderAlgebra`a[_, _] | LadderAlgebra`ad[_, _], Infinity];
    coefficient = term //. {LadderAlgebra`a[_, _] -> 1, LadderAlgebra`ad[_, _] -> 1};
    
    (* Get operator types *)
    type = Head /@ operators;
    
    (* Get indices *)
    downIndices = Level[#,1][[1]]& /@operators; (* First argument of each operator *)
    upIndices = Level[#,1][[2]]& /@operators;   (* Second argument of each operator *)
    
    (* Check that indices form the same set *)
    If[Sort[upIndices] =!= Sort[downIndices],
      Message[identifyTerm::mismatch, downIndices, upIndices];
      Return[$Failed]
    ];
    
    (* Find permutation from down to up indices *)
    sigma = PermutationList[FindPermutation[downIndices, upIndices], Length[type]];
    
    {type, sigma, coefficient}
  ];

identifyTerm::mismatch = "Index mismatch: down indices `1` and up indices `2` don't form the same set.";

IdentifyTerms[expr_] := 
  Module[{terms},
    terms = MonomialList[Expand[expr]];
    identifyTerm /@ terms
  ];


(* ::Subsection::Closed:: *)
(*Term Collection*)


(* Collect terms with a given type *)
collectTypeTerms[identifiedTerms_, type_] := 
  Module[{matchingTerms, permList, permToCoeff},
    
    (* Filter terms matching the type *)
    matchingTerms = Select[identifiedTerms, #[[1]] === type &];
    
    If[Length[matchingTerms] == 0, Return[<||>]];
    
    (* Get unique permutations *)
    permList = Union[#[[2]] & /@ matchingTerms];
    
    (* Initialize association *)
    permToCoeff = AssociationMap[0 &, permList];
    
    (* Aggregate coefficients *)
    Do[
      permToCoeff[term[[2]]] += term[[3]],
      {term, matchingTerms}
    ];
    
    permToCoeff
  ];

CollectTermsByType[identifiedTerms_] := 
  Module[{typeList, typeToPermToCoeff},
    
    (* Get unique types *)
    typeList = Union[#[[1]] & /@ identifiedTerms];
    
    (* Build nested association: type -> {perm -> coeff} *)
    typeToPermToCoeff = AssociationMap[
      collectTypeTerms[identifiedTerms, #] &,
      typeList
    ];
    
    typeToPermToCoeff
  ];


(* ::Subsection::Closed:: *)
(*Simplification via Group Orbits*)


(* Get the symmetry group for a given operator type *)
getTypeGroup[type_] := 
  Module[{transposition, nA, nAd, genA, genB, gens},
    
    (* Helper function for transpositions *)
    transposition[l_Integer, k_Integer, L_Integer] := 
      PermutationList[Cycles[{{l, k}}], L];
    
    (* Count annihilation and creation operators *)
    nA = Count[type, LadderAlgebra`a];
    nAd = Count[type, LadderAlgebra`ad];
    
    (* Generate transpositions for annihilation operators (S_nA) *)
    genA = transposition[#, # + 1, nA + nAd] & /@ Range[nA - 1];
    genA = AppendTo[genA, PermutationList[Cycles[{Range[nA]}], nA + nAd]];
    
    (* Generate transpositions for creation operators (S_nAd) *)
    genB = transposition[#, # + 1, nA + nAd] & /@ Range[nA + 1, nA + nAd - 1];
    genB = AppendTo[genB, PermutationList[Cycles[{Range[nA + 1, nA + nAd]}], nA + nAd]];
    
    (* Combine generators and remove duplicates *)
    gens = Join[genA, genB] // DeleteDuplicates;
    
    PermutationGroup[gens]
  ];


(* Simplify association for a single operator type by grouping permutations in the same orbit *)
simplifyAssoc[type_, collectedTerms_] := 
  Module[{assoc, perms, H, myF, permsCycleForm, orbitsCycleForm, orbits, 
          orbitsPermAssoc, perm, myFunction, newPerms, newCoeff, newAssoc, coeff},
    
    (* Get the association for this type *)
    assoc = collectedTerms[type];
    perms = Keys[assoc];
    
    (* Get the symmetry group for this type *)
    H = getTypeGroup[type];
    
    (* Helper function to convert cycles to permutation lists *)
    myF[list_] := PermutationList[#, Length[type]] & /@ list;
    
    (* Convert permutations to cycle form for group orbit computation *)
    permsCycleForm = PermutationCycles[#] & /@ perms;
    
    (* Compute group orbits (each orbit is a list of permutations equivalent under H) *)
    orbitsCycleForm = GroupOrbits[H, {#}][[1]] & /@ permsCycleForm;
    orbits = myF /@ orbitsCycleForm;
    
    (* Map each permutation to its orbit *)
    orbitsPermAssoc = AssociationThread[perms, orbits];
    
    (* Initialize new collection *)
    newPerms = {};
    newCoeff = {};
    
    (* Process permutations, grouping by orbits *)
    While[Length[perms] > 0,
      (* Take the first permutation *)
      perm = First[perms];
      perms = Drop[perms, 1];
      
      (* Add it as a representative *)
      AppendTo[newPerms, perm];
      coeff = assoc[perm];
      
      (* Function to check if another permutation is in the same orbit *)
      myFunction[permTest_] := 
        If[orbitsPermAssoc[perm] == orbitsPermAssoc[permTest],
          (* Same orbit: add coefficient and remove from list *)
          perms = DeleteElements[perms, {permTest}];
          coeff += assoc[permTest];,
          (* Different orbit: do nothing *)
          Null;
        ];
      
      (* Apply to all remaining permutations *)
      myFunction /@ perms;
      
      (* Store the summed coefficient *)
      AppendTo[newCoeff, coeff];
    ];
    
    (* Build new association *)
    newAssoc = AssociationThread[newPerms, newCoeff];
    
    newAssoc
  ];


(* Main simplification function: simplify all types *)
SimplifyCollectedTerms::usage = 
  "SimplifyCollectedTerms[collectedTerms] simplifies the collected terms by grouping \
permutations that are equivalent under the symmetry group S_nA \[Times] S_nAd for each operator type.";

SimplifyCollectedTerms[collectedTerms_] := 
  Module[{types, newAssocs, newCollection},
    
    (* Get all operator types *)
    types = Keys[collectedTerms];
    
    (* Simplify each type *)
    newAssocs = simplifyAssoc[#, collectedTerms] & /@ types;
    
    (* Build new collection *)
    newCollection = AssociationThread[types, newAssocs];
    
    newCollection
  ];


(* ::Subsection::Closed:: *)
(*Reconstruction*)


(* Reconstruct a generalized trace from type and permutation *)
reconstructGeneralizedTrace[type_, sigma_] := 
  Module[{sigmaUpdated, downIndices, upIndices, terms},
    
    (* Handle identity permutation *)
    sigmaUpdated = If[Length[sigma] == 0,
      Table[i, {i, 1, Length[type]}],
      sigma
    ];
    
    If[Length[type] =!= Length[sigmaUpdated],
      Message[reconstructGeneralizedTrace::length];
      Return[$Failed]
    ];
    
    (* Create index symbols *)
    downIndices = Table[Symbol["k" <> ToString[i]], {i, 1, Length[sigmaUpdated]}];
    upIndices = downIndices[[sigmaUpdated]];
    
    (* Build operator product *)
    terms = MapThread[#1[#2, #3] &, {type, downIndices, upIndices}];
    
    Distribute[Apply[NonCommutativeMultiply, terms]]
  ];

reconstructGeneralizedTrace::length = "Length mismatch between type and permutation.";

(* Sum over all types and permutations *)
ReconstructTrace[typeToPermToCoeff_Association] := 
  Module[{typeTerms},
    typeTerms = KeyValueMap[
      Function[{type, permToCoeff},
        Total[
          KeyValueMap[
            #2 * reconstructGeneralizedTrace[type, #1] &,
            permToCoeff
          ]
        ]
      ],
      typeToPermToCoeff
    ];
    
    Total[typeTerms]
  ];


(* ::Subsection::Closed:: *)
(*Main Decomposition Pipeline*)


Options[DecomposeTrace] = {
  Verbose -> False,
  PrintTable -> False,
  Simplify -> True
};

(* Default mass parameter *)
DecomposeTrace[ops_String, perm_List, opts : OptionsPattern[]] := 
  DecomposeTrace[ops, perm, 1, opts];

(* Convenience: integer K for backward-compat during transition *)
DecomposeTrace[n_Integer, mass_, opts : OptionsPattern[]] := 
  DecomposeTrace[
    StringJoin[Table["X", n]],
    PermutationList[Cycles[{Range[n]}], n],
    mass, opts
  ];

DecomposeTrace[n_Integer, opts : OptionsPattern[]] := 
  DecomposeTrace[n, 1, opts];

(* Main implementation *)
DecomposeTrace[ops_String, perm_List, mass_, opts : OptionsPattern[]] := 
  Module[{verbose, printTable, simplify, expr, expandedExpr, orderedExpr, 
          contractedExpr, identifiedTerms, collectedTerms},
    
    (* Parse options *)
    verbose = OptionValue[Verbose];
    printTable = OptionValue[PrintTable];
    simplify = OptionValue[Simplify];
    
    (* Pipeline steps *)
    expr = MonomialTrace[ops, perm, mass];
    expandedExpr = Distribute[expr];
    orderedExpr = antiNormalOrder[expandedExpr];
    contractedExpr = ContractIndices[orderedExpr];
    identifiedTerms = IdentifyTerms[contractedExpr];
    collectedTerms = CollectTermsByType[identifiedTerms];
    
    (* Apply simplification via group orbits if requested *)
    If[simplify,
      collectedTerms = SimplifyCollectedTerms[collectedTerms];
    ];
    
    (* Verbose output *)
    If[verbose,
      Print["Monomial trace: ", TraditionalForm[expr]];
      Print["Anti-normal ordered: ", TraditionalForm[orderedExpr]];
      Print["Contracted: ", TraditionalForm[contractedExpr]];
      If[simplify,
        Print["Simplified by grouping equivalent permutations."];
      ];
    ];
    
    (* Print table *)
    If[printTable,
      Print["Decomposition of Tr_perm(", ops, "):"];
      Print["=" <> StringRepeat["=", 50]];
      printDecompositionTable[collectedTerms];
    ];
    
    collectedTerms
  ];

(* Helper function to print decomposition as a table *)
printDecompositionTable[collectedTerms_Association] := 
  Module[{rows, header},
    header = {"Type", "Permutation", "Coefficient"};
    
    rows = Flatten[
      KeyValueMap[
        Function[{type, permToCoeff},
          MapIndexed[
            Function[{permCoeffPair, idx},
              If[idx[[1]] == 1,
                {type, permCoeffPair[[1]], permCoeffPair[[2]]},
                {Null, permCoeffPair[[1]], permCoeffPair[[2]]}
              ]
            ],
            Normal[permToCoeff]
          ]
        ],
        collectedTerms
      ],
      1
    ];
    
    Print[Grid[
      Prepend[rows, header],
      Frame -> All,
      Background -> {None, {1 -> LightBlue}},
      Alignment -> {Left, Center}
    ]];
  ];


(* ::Section::Closed:: *)
(*Overlap Calculations with Young Projectors*)


(* ::Subsection::Closed:: *)
(*Young Projector Symbols*)


(* Declare Young projector with operators: P_label[downIndices, upIndices] * (products of operators) *)
DeclareYoungProjector::usage = 
  "DeclareYoungProjector[n, isLeft] creates a Young projector symbol with n operators. \
If isLeft=True, uses annihilation operators (left state). If isLeft=False, uses creation operators (right state).";

DeclareYoungProjector[n_Integer, isLeft_?BooleanQ] := 
  Module[{downIndices, upIndices, ops, label},
    If[isLeft,
      (* Left state: use annihilation operators with i,j indices *)
      downIndices = Table[Symbol["i" <> ToString[i]], {i, 1, n}];
      upIndices = Table[Symbol["j" <> ToString[i]], {i, 1, n}];
      label = "\[ScriptL]";
      ops = MapThread[LadderAlgebra`a, {upIndices, downIndices}],
      
      (* Right state: use creation operators with m,n indices *)
      downIndices = Table[Symbol["m" <> ToString[i]], {i, 1, n}];
      upIndices = Table[Symbol["n" <> ToString[i]], {i, 1, n}];
      label = "r";
      ops = MapThread[LadderAlgebra`ad, {upIndices, downIndices}]
    ];
    
    (* Return P_label[downIndices, upIndices] * (product of operators) *)
    Subscript[P, label][downIndices, upIndices] * Apply[NonCommutativeMultiply, ops]
  ];

(* Formatting for Young projectors *)
Format[Subscript[P, s_String][downIndices_List, upIndices_List], TraditionalForm] := 
  Module[{subI, supJ},
    subI = Row[Riffle[downIndices, ","]];
    supJ = Row[Riffle[upIndices, ","]];
    Subsuperscript[Subscript[P, s, ""], subI, supJ]
  ];

(* Make Young projectors behave as scalars in NonCommutativeMultiply *)
Subscript /: NumericQ[Subscript[P, _][__]] = True;
Subscript /: NonCommutativeMultiply[pre___, c : Subscript[P, _][__], post___] := 
  c * NonCommutativeMultiply[pre, post];
Subscript /: NonCommutativeMultiply[c : Subscript[P, _][__]] := c;


(* ::Subsection::Closed:: *)
(*Wick's Formula Implementation*)


(* Extract indices from operators *)
extractOperatorIndices[expr_] := 
  Module[{aOps, adOps, iIndices, jIndices},
    (* Extract all annihilation and creation operators *)
    aOps = Cases[expr, LadderAlgebra`a[i_, j_] :> {i, j}, Infinity];
    adOps = Cases[expr, LadderAlgebra`ad[i_, j_] :> {i, j}, Infinity];
    
    (* Separate into lower and upper indices *)
    iIndices = Join[aOps[[All, 1]], adOps[[All, 1]]];
    jIndices = Join[aOps[[All, 2]], adOps[[All, 2]]];
    
    {iIndices, jIndices}
  ];

(* Create Kronecker delta product for a permutation in Wick's theorem *)
createWickDeltas[iIndicesA_List, jIndicesA_List, iIndicesAd_List, jIndicesAd_List, sigma_List] := 
  Module[{n, sigmaInv, deltaLower, deltaUpper},
    n = Length[sigma];
    sigmaInv = InversePermutation[sigma];
    
    (* Lower index deltas: \[Delta]^(l_{\[Sigma]^{-1}(s)})_(i_s) *)
    deltaLower = Product[
      LadderAlgebra`\[Delta][iIndicesA[[s]], jIndicesAd[[sigma[[s]]]]],
      {s, 1, n}
    ];
    
    (* Upper index deltas: \[Delta]_(k_s)^(j_{\[Sigma](s)}) *)
    deltaUpper = Product[
      LadderAlgebra`\[Delta][iIndicesAd[[s]], jIndicesA[[sigmaInv[[s]]]]],
      {s, 1, n}
    ];
    
    deltaLower * deltaUpper
  ];

(* Apply Wick's theorem to contract operators *)
wickContract[expr_] := 
  Module[{aOps, adOps, nA, nAd, iIndicesA, jIndicesA, iIndicesAd, jIndicesAd, result},
    (* Extract operators *)
    aOps = Cases[expr, LadderAlgebra`a[_, _], Infinity];
    adOps = Cases[expr, LadderAlgebra`ad[_, _], Infinity];
    nA = Length[aOps];
    nAd = Length[adOps];
    
    (* Check if number of a and ad operators match *)
    If[nA =!= nAd, Return[0]];
    
    (* Extract indices *)
    iIndicesA = aOps[[All, 1]];
    jIndicesA = aOps[[All, 2]];
    iIndicesAd = adOps[[All, 1]];
    jIndicesAd = adOps[[All, 2]];
    
    (* Apply Wick's formula: sum over all permutations *)
    result = Sum[
      createWickDeltas[iIndicesA, jIndicesA, iIndicesAd, jIndicesAd, PermutationList[sigma, nA]],
      {sigma, Permutations[Range[nA]]}
    ];
    
    result
  ];


(* Load pre-computed character table for S_n from file *)
LoadDoubleCosetsRepsAndSizes::usage = 
  "LoadDoubleCosetsRepsAndSizes[p, n, q] loads the representatives and sizes of the double cosets Sp s Sq where s is in S_n.";

With[{dir = $PackageRoot},
    LoadDoubleCosetsRepsAndSizes[p_, n_, q_] := LoadDoubleCosetsRepsAndSizes[p, n, q] = Association[
        Get[
            FileNameJoin[{dir, "data/processed/double_cosets/", StringJoin["dc_n", ToString[n], "_p", ToString[p], "_q", ToString[q], ".txt"]}]
        ]
    ];
];

(* Apply Wick's theorem to contract operators *)
wickContractEfficient[expr_, nLeft_Integer, nRight_Integer] := 
  Module[{aOps, adOps, nA, nAd, iIndicesA, jIndicesA, iIndicesAd, jIndicesAd, dcSizesList, dcRepsList, result},
    (* Extract operators *)
    aOps = Cases[expr, LadderAlgebra`a[_, _], Infinity];
    adOps = Cases[expr, LadderAlgebra`ad[_, _], Infinity];
    nA = Length[aOps];
    nAd = Length[adOps];

    (* Check if number of a and ad operators match *)
    If[nA =!= nAd, Return[0]];

	(* Get the double cosets representatives and sizes *)
	dcSizesList = LoadDoubleCosetsRepsAndSizes[nLeft,nA,nRight]["Sizes"];
	dcRepsList = LoadDoubleCosetsRepsAndSizes[nLeft,nA,nRight]["Representatives"];
    
    (* Extract indices *)
    iIndicesA = aOps[[All, 1]];
    jIndicesA = aOps[[All, 2]];
    iIndicesAd = adOps[[All, 1]];
    jIndicesAd = adOps[[All, 2]];
    
    (* Apply Wick's formula: sum over all permutations *)
    result = Total@MapThread[#2*createWickDeltas[iIndicesA, jIndicesA, iIndicesAd, jIndicesAd, #1]&,{dcRepsList, dcSizesList}];
    
    result
  ];


ApplyWickTheorem::usage = 
  "ApplyWickTheorem[expr] applies Wick's theorem to contract all operators, \
replacing them with appropriate Kronecker deltas.";

ApplyWickTheorem[expr_, nLeft_:-1, nRight_:-1] := 
  Module[{wickResult, scalarPart},
    (* Apply Wick contraction to get delta structure *)
    If[nLeft==-1,
      wickResult = wickContract[expr];,
      wickResult = wickContractEfficient[expr, nLeft, nRight];
    ];
    
    (* Extract scalar part (everything except operators) *)
    scalarPart = expr //. {LadderAlgebra`a[_, _] -> 1, LadderAlgebra`ad[_, _] -> 1};
    
    (* Multiply and expand *)
    Expand[wickResult * scalarPart]
  ];


(* ::Subsection::Closed:: *)
(*Index Relabeling for Young Projectors*)


(*(* Extract combined down indices from Young projector products *)
extractCombinedDownIndices[c_ * Subscript[P, R_][downA_List, upA_List] * 
                            Subscript[P, S_][downB_List, upB_List]] := 
  Join[downA, downB];
  
extractCombinedDownIndices[Subscript[P, R_][downA_List, upA_List] * 
                            Subscript[P, S_][downB_List, upB_List]] := 
  Join[downA, downB];

(* Fallback: no Young projectors means no indices *)
extractCombinedDownIndices[_] := {};

(* Extract combined up indices from Young projector products *)
extractCombinedUpIndices[c_ * Subscript[P, R_][downA_List, upA_List] * 
                          Subscript[P, S_][downB_List, upB_List]] := 
  Join[upA, upB];

extractCombinedUpIndices[Subscript[P, R_][downA_List, upA_List] * 
                          Subscript[P, S_][downB_List, upB_List]] := 
  Join[upA, upB];

(* Fallback: no Young projectors means no indices *)
extractCombinedUpIndices[_] := {};*)

(* Extract combined down indices from Young projector products *)
extractCombinedDownIndices[c_ * Subscript[P, "\[ScriptL]"][downA_List, upA_List] * 
                            Subscript[P, "r"][downB_List, upB_List]] := 
  Join[downA, downB];

extractCombinedDownIndices[c_ * Subscript[P, "r"][downB_List, upB_List] * 
                            Subscript[P, "\[ScriptL]"][downA_List, upA_List]] := 
  Join[downA, downB];
  
extractCombinedDownIndices[Subscript[P, "\[ScriptL]"][downA_List, upA_List] * 
                            Subscript[P,"r"][downB_List, upB_List]] := 
  Join[downA, downB];

extractCombinedDownIndices[Subscript[P, "r"][downB_List, upB_List] * 
                            Subscript[P, "\[ScriptL]"][downA_List, upA_List]] := 
  Join[downA, downB];

(* Fallback: no Young projectors means no indices *)
extractCombinedDownIndices[_] := {};

(* Extract combined up indices from Young projector products *)
extractCombinedUpIndices[c_ * Subscript[P, "\[ScriptL]"][downA_List, upA_List] * 
                            Subscript[P, "r"][downB_List, upB_List]] := 
  Join[upA, upB];

extractCombinedUpIndices[c_ * Subscript[P, "r"][downB_List, upB_List] * 
                            Subscript[P, "\[ScriptL]"][downA_List, upA_List]] := 
  Join[upA, upB];
  
extractCombinedUpIndices[Subscript[P, "\[ScriptL]"][downA_List, upA_List] * 
                            Subscript[P,"r"][downB_List, upB_List]] := 
  Join[upA, upB];

extractCombinedUpIndices[Subscript[P, "r"][downB_List, upB_List] * 
                            Subscript[P, "\[ScriptL]"][downA_List, upA_List]] := 
  Join[upA, upB];

(* Fallback: no Young projectors means no indices *)
extractCombinedUpIndices[_] := {};

(* Relabel indices in a single term *)
relabelTermIndices[term_] := 
  Module[{indexList, newIndices, replacementRule},
   (* Sort the term first to ensure P_\[ScriptL] comes before P_r *)
    indexList = extractCombinedDownIndices[term];
    newIndices = Table[Symbol["k" <> ToString[i]], {i, 1, Length[indexList]}];
    replacementRule = AssociationThread[indexList, newIndices];
    term /. replacementRule
  ];

RelabelIndices::usage = 
  "RelabelIndices[expr] relabels all indices with k1, k2, etc. using canonical ordering based on down indices.";

RelabelIndices[expr_] := 
  Module[{terms},
    terms = MonomialList[Expand[expr]];
    Total[relabelTermIndices /@ terms] // Simplify
  ];


(* ::Subsection::Closed:: *)
(*Abstract Tensor Contractions*)


(* Get contraction pairs from a single term *)
getTermContractionPairs[term_] := 
  Module[{downIndices, upIndices, perm},
    downIndices = extractCombinedDownIndices[term];
    upIndices = extractCombinedUpIndices[term];
    perm = PermutationList[FindPermutation[downIndices, upIndices], Length[downIndices]];
    Table[{i, perm[[i]]+Length[perm]},{i, 1, Length[perm]}]
  ];

(* Replace Young projector product by abstract tensor contraction *)
replaceByTensorContraction[term_] := 
  Module[{coeff, pair},
    pair = getTermContractionPairs[term];
    coeff = term //. Subscript[P, _][__] -> 1;
    coeff * TensorContract[Subscript[P, "\[ScriptL]"] \[TensorProduct] Subscript[P, "r"], pair]
  ];

ConvertToTensorContractions::usage = 
  "ConvertToTensorContractions[expr] converts Young projector products into abstract tensor contractions.";

ConvertToTensorContractions[expr_] := 
  Module[{terms},
    terms = MonomialList[expr];
    Total[replaceByTensorContraction /@ terms]
  ];


(* ::Subsection::Closed:: *)
(*Canonicalization using TensorReduce*)


(* Generate symmetry generators for Young projector tensors *)
getSymmetryGenerators[K_Integer] := 
  Module[{transposition, gens, inLegs, outLegs, sym},
    (* Transposition generator: swap adjacent pairs (l, l+1) and (l+K, l+K+1) *)
    transposition[l_, L_] := 
      PermutationList[Cycles[{{l, l + 1}, {l + L, l + L + 1}}], 2 * L];
    
    (* Generate adjacent transpositions *)
    gens = transposition[#, K] & /@ Table[l, {l, 1, K - 1}];
    
    (* Add exchange of in-legs and out-legs *)
    inLegs = Table[l, {l, 1, K}];
    outLegs = Table[l, {l, K + 1, 2 * K}];
    AppendTo[gens, PermutationList[Cycles[{inLegs, outLegs}]]];
    
    (* Remove duplicates and format for TensorReduce *)
    gens = DeleteDuplicates[gens];
    sym = {#, 1} & /@ gens;
    
    sym
  ];

CanonicalizeTensorExpression::usage = 
  "CanonicalizeTensorExpression[expr, nLeft, nRight] applies TensorReduce to canonicalize \
tensor contraction expressions with left and right Young projectors of sizes nLeft and nRight.";

CanonicalizeTensorExpression[expr_, nLeft_Integer, nRight_Integer] := 
  Module[{symLeft, symRight, dimsLeft, dimsRight, assumptions},
    
    (* Get symmetry generators *)
    symLeft = getSymmetryGenerators[nLeft];
    symRight = getSymmetryGenerators[nRight];
    
    (* Define dimensions *)
    dimsLeft = Table[d, {i, 1, 2 * nLeft}];
    dimsRight = Table[d, {i, 1, 2 * nRight}];
    
    (* Set up assumptions for TensorReduce *)
    assumptions = {
      d \[Element] Reals,
      Subscript[P, "\[ScriptL]"] \[Element] Arrays[dimsLeft, Complexes, symLeft],
      Subscript[P, "r"] \[Element] Arrays[dimsRight, Complexes, symRight]
    };
    
    (* Apply TensorReduce with assumptions *)
    Assuming[assumptions, Factor[TensorReduce[expr]]]
  ];


(* ::Subsection:: *)
(*Main Overlap Calculation Pipeline*)


ComputeOverlap::usage = 
  "ComputeOverlap[nLeft, type, sigma, opts] computes the overlap matrix element \
\[LeftAngleBracket]nLeft|Tr_\[Sigma](type)|nRight\[RightAngleBracket] where nRight is determined by excitation balance. \
Options: Verbose->False, Efficient->True.";

Options[ComputeOverlap] = {
  Verbose -> False,
  Efficient -> True
};

ComputeOverlap[nLeft_Integer, type_List, sigma_List, opts : OptionsPattern[]] := 
  Module[{verbose, efficient, nRight, expr, wickExpanded, contracted, 
          relabeled, tensorForm, nTerms},
    
    (* Parse options *)
    verbose = OptionValue[Verbose];
    efficient = OptionValue[Efficient];
    
    (* Determine right excitation number (balance equation) *)
    nRight = nLeft + Count[type, LadderAlgebra`a] - Count[type, LadderAlgebra`ad];
    
    If[nRight < 0,
      Message[ComputeOverlap::negative, nRight];
      Return[$Failed]
    ];
    
    (* Build overlap expression: \[LeftAngleBracket]left| Tr_\[Sigma](type) |right\[RightAngleBracket] *)
    expr = DeclareYoungProjector[nLeft, True] ** DeclareYoungProjector[nRight, False] ** LadderAlgebra`trSigma[PermutationCycles[sigma], Sequence @@ type];
    
    (* Pipeline steps *)
    If[efficient,
      wickExpanded = ApplyWickTheorem[expr, nLeft, nRight];,
      wickExpanded = ApplyWickTheorem[expr];
    ];
    contracted = ContractIndices[wickExpanded];
    relabeled = RelabelIndices[contracted];
    tensorForm = ConvertToTensorContractions[relabeled];
    
    (* Count terms for ratio *)
    nTerms = Length[Cases[tensorForm, TensorContract[_, _], {0,Infinity}]];
    
    (* Verbose output *)
    If[verbose,
      Print["Overlap expression: ", TraditionalForm[expr]];
      Print["After Wick's theorem: ", TraditionalForm[wickExpanded]];
      Print["Contracted indices: ", TraditionalForm[contracted]];
      Print["Relabeled: ", TraditionalForm[relabeled]];
      Print["Tensor contractions: ", TraditionalForm[tensorForm]];
      Print["Number of terms: ", nTerms];
    ];
    
    tensorForm
  ];

ComputeOverlap::negative = "Right excitation number `1` is negative. This overlap vanishes.";


(* ::Section::Closed:: *)
(*Matrix Element Exact Calculations with Character Theory*)


(* ::Subsection::Closed:: *)
(*Helper Functions*)


(* ::Subsubsection::Closed:: *)
(*Permutation Helpers*)


(* Tensor product of two permutations: extends the second permutation to act on disjoint indices *)
PermutationTensorProduct::usage = 
  "PermutationTensorProduct[sigma1, sigma2] computes the tensor product of two permutations.";

PermutationTensorProduct[sigma1_List, sigma2_List] := 
  Join[sigma1, sigma2 + Length[sigma1]];


(* Create a permutation in cycle form from an integer partition (conjugacy class representative) *)
CreateCyclesFromPartition::usage = 
  "CreateCyclesFromPartition[partition] creates a permutation in cycle form from a given integer partition.";

CreateCyclesFromPartition[partition_List] := 
  Module[{n, elements},
    n = Total[partition];
    elements = Range[n];
    Cycles[TakeList[elements, partition]]
  ];


(* ::Subsubsection::Closed:: *)
(*Theory Quantities*)


(* Product of all hook lengths in a Young tableau *)
HookLengthsProduct[R_] := 
  Apply[Times, ResourceFunction["HookLengths"][R], {0, 1}];

(* Convert partition to list of box coordinates {i, j} *)
PartitionToBoxes[R_] := 
  Flatten[Table[{i, j}, {i, Length[R]}, {j, R[[i]]}], 1];

(* Dimension of the irreducible representation R for SU(d) *)
DimR[R_] := 
  Apply[Times, d + #2 - #1 & @@@ PartitionToBoxes[R]] / HookLengthsProduct[R];

(* Dimension of the irreducible representation R for S_n *)
dimR[R_] := 
  Factorial[Total[R]] / HookLengthsProduct[R];

(* Normalization factor for Young projector states *)
Normalization[R_] := 
  Apply[Times, d + #2 - #1 & @@@ PartitionToBoxes[R]];


(* ::Subsubsection::Closed:: *)
(*Interface with GAP Data*)


(* Load pre-computed character table for S_n from file *)
LoadCharacterTable::usage = 
  "LoadCharacterTable[n] loads the character table for the symmetric group S_n from data files.";

With[{dir = $PackageRoot},
    LoadCharacterTable[n_] := LoadCharacterTable[n] = Association[
        Get[
            FileNameJoin[{dir, "data/processed/conjugacy_classes/", StringJoin["ssct_", ToString[n], ".txt"]}]
        ]
    ];
];


(* ::Subsection::Closed:: *)
(*Permutation Tensor Contractions*)


ContractPermutationTensorOnePair::usage = 
  "ContractPermutationTensorOnePair[sigma, pair, n] contracts a rank-(n,n) permutation tensor \
according to a single pairing pattern.";

ContractPermutationTensorOnePair[sigma_List, pair_List, n_Integer] := 
  Module[{indices, lowerIndices, upperIndices, tensor, contractedExpr},
    
    (* Step 1: Generate 2n unique symbolic indices for tensor legs *)
    (*indices = Table[Unique["k"], {2*n}];*)
    indices = Table[Symbol["k"<>ToString[i]], {i,2*n}];
    lowerIndices = Take[indices, n];
    upperIndices = Take[indices, -n];
    
    (* Step 2: Represent the permutation tensor as a product of Kronecker deltas *)
    (* Mathematical form: (P_sigma)_{i1,...,in}^{j1,...,jn} = \[Delta]_{i1}^{j_\[Sigma](1)} * ... * \[Delta]_{in}^{j_\[Sigma](n)} *)
    tensor = Times @@ Table[
      LadderAlgebra`\[Delta][lowerIndices[[k]], upperIndices[[sigma[[k]]]]],
      {k, 1, n}
    ];
    
    (* Step 3: Apply contraction by substituting paired indices *)
    (* For each pair {p1, p2}, set indices[[p2]] = indices[[p1]] *)
    contractedExpr = Fold[
      #1 /. indices[[#2[[2]]]] -> indices[[#2[[1]]]] &,
      tensor,
      pair
    ];
    
    (* Step 4: Simplify and contract the remaining Kronecker deltas *)
    ContractIndices[contractedExpr]
  ];


ContractPermutationTensor::usage = 
  "ContractPermutationTensor[sigma, pairs, n] computes the sum of contractions for multiple pairing patterns.";

ContractPermutationTensor[sigma_List, pairs_List, n_Integer] := 
  Module[{result},
    (* Sum contractions over all pairing patterns *)
    result = Simplify[Total[ContractPermutationTensorOnePair[sigma, #, n] & /@ pairs]];
    
    (* Debug output (can be removed in production) *)
    Echo[sigma, "sigma"];
    Echo[pairs, "pairs"];
    Echo[n, "n"];
    Echo[result, "result"];
    
    result
  ];


(* ::Subsection::Closed:: *)
(*Extract Coefficients and Pairs from Canonicalized Expressions*)


GetCoefficientAndPairs::usage = 
  "GetCoefficientAndPairs[term] extracts the coefficient and contraction pairs from a TensorContract expression.";

GetCoefficientAndPairs[term_] := 
  Module[{coeff, pairs},
    Which[
      (* Case 1: Term is coefficient times TensorContract *)
      MatchQ[term, _ * TensorContract[_, _]],
        coeff = term /. TensorContract[_, _] -> 1;
        pairs = Cases[term, TensorContract[_, p_] :> p, Infinity][[1]];
        {pairs, coeff},
      
      (* Case 2: Term is just TensorContract (coefficient = 1) *)
      MatchQ[term, TensorContract[_, _]],
        pairs = Cases[term, TensorContract[_, p_] :> p, {0, Infinity}][[1]];
        {pairs, 1},
      
      (* Case 3: Term is just a coefficient (no contraction) *)
      True,
        {{}, term}
    ]
  ];


ExtractTerms::usage = 
  "ExtractTerms[expr] extracts individual terms from a sum, treating TensorContract as atomic.";

ExtractTerms[expr_] := 
  Module[{factored, termList},
    
    (* Factor the expression treating TensorContract as atomic *)
    factored = Collect[expr, _TensorContract, Simplify];
    
    (* Extract terms based on expression head *)
    termList = Which[
      (* Case 1: Expression is a sum of terms *)
      Head[factored] === Plus,
        List @@ factored,
      
      (* Case 2: Expression is a single term *)
      True,
        {factored}
    ];
    
    termList
  ];


GetPairsAndCoefficientsAssociation::usage = 
  "GetPairsAndCoefficientsAssociation[expr] extracts all contraction patterns and their coefficients \
from a canonicalized tensor expression, returning an Association mapping pairs -> coefficients.";

GetPairsAndCoefficientsAssociation[expr_] := 
  Module[{terms, pairsAndCoeffs, result},
    
    (* Check if expression contains any tensor contractions *)
    If[Length[Cases[expr, TensorContract[_, _], Infinity]] == 0,
      (* No tensor contractions present: expression is just a coefficient *)
      terms = {expr},
      
      (* Extract all terms from the sum *)
      terms = ExtractTerms[expr]
    ];
    
    (* Extract pairs and coefficients from each term *)
    pairsAndCoeffs = Transpose[GetCoefficientAndPairs /@ terms];
    
    (* Build association mapping contraction patterns to coefficients *)
    result = AssociationThread[pairsAndCoeffs[[1]] -> pairsAndCoeffs[[2]]];
    
    result
  ];


(* ::Subsection::Closed:: *)
(*Character Theory Contractions*)


GetRepresentationIndex::usage = 
  "GetRepresentationIndex[partition] finds the 1-based index of the irreducible representation \
corresponding to a partition in S_n.";

GetRepresentationIndex[R_List] := 
  Module[{n, repNames, idx},
    
    (* Determine the size of the partition *)
    n = Total[R];
    
    (* Get list of representation labels (partitions) from character table *)
    repNames = LoadCharacterTable[n]["CharacterParameters"];
    
    (* Find the position of this partition in the list *)
    idx = FirstPosition[repNames, R];
    
    (* Handle case where partition is not found *)
    If[idx === Missing["NotFound"],
      Message[GetRepresentationIndex::notfound, R, n];
      Return[$Failed],
      
      First[idx]
    ]
  ];

GetRepresentationIndex::notfound = "Partition `1` not found in S_`2` representations.";


(* ::Subsubsection::Closed:: *)
(*Computation using decompositions into tensor contractions*)


ComputeContractions::usage = 
  "ComputeContractions[R, S, pairs] computes the sum of tensor contractions weighted by \
character theory coefficients for partitions R (left) and S (right).";

ComputeContractions[R_List, S_List, pairs_List] := 
Module[
  {nR,nS,indexRepR,indexRepS,charTableR,charTableS,nbConjClR,nbConjClS,conjClR,conjClS,sizeConjClR,sizeConjClS,elementsR,elementsS,A,Ad,myFunction,myCoeff,result},
    
   (* Partition sizes *)
    nR = Total[R];
    nS = Total[S];

    (* Find representation indices *)
    indexRepR = GetRepresentationIndex[R];
    indexRepS = GetRepresentationIndex[S];
    
    (* Get character tables *)
    charTableR = LoadCharacterTable[nR]["CharacterTable"];
    charTableS = LoadCharacterTable[nS]["CharacterTable"];

	(* Conjugacy classes *)
	conjClR = LoadCharacterTable[nR]["CharacterParameters"];
	conjClS = LoadCharacterTable[nS]["CharacterParameters"];

    (* Number of conjugacy classes *)
    nbConjClR = Length[conjClR];
    nbConjClS = Length[conjClS];
	
	(* Get elements in each conjugacy classes *)
	elementsR = LoadCharacterTable[nR]["ElementsConjugacyClasses"];
	elementsS = LoadCharacterTable[nS]["ElementsConjugacyClasses"];
    
    (* Function to sum contractions over conjugacy class pairs *)
    myFunction[iR_, iS_] := 
      Module[{sigmaRSList},
        sigmaRSList =  PermutationTensorProduct[#1, #2] & @@@Tuples[{elementsR[[iR]], elementsS[[iS]]}];
        Echo[sigmaRSList,"sigmaRSList"];
        Total[ContractPermutationTensor[#, pairs, nR + nS] & /@ sigmaRSList]
      ];
    
    (* Character theory coefficient *)
      myCoeff[iR_, iS_] := 
      charTableR[[indexRepR, iR]] * charTableS[[indexRepS, iS]] / (Factorial[nR] Factorial[nS]);
    
    (* Main sum over conjugacy classes *)
    result = Sum[
      myCoeff[iR, iS] * myFunction[iR, iS],
      {iR, 1, nbConjClR},
      {iS, 1, nbConjClS}
    ];
    
    result
  ];


(* ::Subsection:: *)
(*Single Overlap Computation*)


ComputeSingleOverlap::usage = 
  "ComputeSingleOverlap[R, S, type, sigma, opts] computes a single overlap matrix element \
\[LeftAngleBracket]R|Tr_\[Sigma](type)|S\[RightAngleBracket] where R and S are partitions. Options: Verbose->False.";

Options[ComputeSingleOverlap] = {
  Verbose -> False,
  Efficient -> True
};

ComputeSingleOverlap[R_List, S_List, type_List, sigma_List, opts : OptionsPattern[]] := 
  Module[{verbose, efficient, nL, nR, check, canonicalizedExpr, pairsAndCoeffsAssociation, result},
    
    verbose = OptionValue[Verbose];
    efficient = OptionValue[Efficient];
    
    If[verbose,
      Print["-------------------------"];
      Print["Enter \"ComputeSingleOverlap\" function"];
      Print["Compute single overlap for: R=", R,", S=", S,", type=", type,", sigma=", sigma];
    ];
    
    (* Partition sizes *)
    nL = Total[R];
    nR = Total[S];
    
    (* Check excitation balance *)
    check = nR =!= nL + Count[type, LadderAlgebra`a] - Count[type, LadderAlgebra`ad];
    
    If[check,
      (* Excitations don't balance, overlap vanishes *)
      If[verbose, Print["Excitation mismatch: overlap vanishes"]];
      Return[0],
      
      (* Compute overlap using canonicalized tensor expression *)
      canonicalizedExpr = ComputeOverlap[nL, type, sigma, Verbose -> False, Efficient -> efficient];
      pairsAndCoeffsAssociation = GetPairsAndCoefficientsAssociation[canonicalizedExpr];
      
      (* Sum contractions weighted by coefficients *)
      result = Total[
        KeyValueMap[
          #2 * ComputeContractions[R, S, {#1}] &,
          pairsAndCoeffsAssociation
        ]
      ];
      
      (* Handle vacuum state case (remove projector symbols) *)
      result = result /. {Subscript[P, "r"] | Subscript[P, "\[ScriptL]"] -> 1} // Simplify;
      
      (* Normalize *)
      result = result/Sqrt[Normalization[R]Normalization[S]];
      
      If[verbose,
        Print["Canonicalized expression: ", canonicalizedExpr];
        Print["Pairs and coefficients: ", pairsAndCoeffsAssociation];
        Print["Result: ", result];
        Print["-------------------------"];
      ];
      
      result
    ]
  ];


(* ::Subsection::Closed:: *)
(*Main Matrix Element Computation*)


ComputeMatrixElementInteraction::usage = 
  "ComputeMatrixElementInteraction[R, S, ops, perm, opts] computes the matrix element \
\[LeftAngleBracket]R|Tr_perm(M(ops))|S\[RightAngleBracket] by decomposing the monomial and summing over all overlap contributions. \
Options: Verbose->False, Efficient->True, UseWovenContractions->True.";

Options[ComputeMatrixElementInteraction] = {
  Verbose -> False,
  Efficient -> True,
  UseWovenContractions -> True
};

ComputeMatrixElementInteraction[R_List, S_List, ops_String, perm_List, opts : OptionsPattern[]] := 
  Module[{verbose, efficient, useWoven, nL, nR, Lambda, wovenData, wovenContractions,
          collectedTerms, sumByType, matrixElement},
    
    verbose = OptionValue[Verbose];
    efficient = OptionValue[Efficient];
    useWoven = OptionValue[UseWovenContractions];
    
    If[verbose,
      Print["------------------------------------"];
      Print["Enter \"ComputeMatrixElementInteraction\" function"];
      Print["Computing matrix element for R=", R, ", S=", S, ", ops=", ops, ", perm=", perm];
    ];
    
    If[useWoven,
      (* === Optimized path: use pre-computed woven contractions === *)
      nL = Total[R];
      nR = Total[S];
      Lambda = Max[nL, nR];
      
      If[verbose, Print["Using woven contractions with Lambda=", Lambda]];
      
      (* Import (or compute+cache) woven contractions *)
      wovenData = ImportWovenContractions[ops, perm, Lambda];
      
      (* Look up the entry for this excitation level pair *)
      wovenContractions = wovenData[{nL, nR}];
      
      If[MissingQ[wovenContractions],
        (* No contributing contractions for this {nL, nR} pair *)
        If[verbose, Print["No woven contractions for {nL, nR}=", {nL, nR}, ". Overlap vanishes."]];
        Return[0]
      ];
      
      If[verbose, Print[Length[wovenContractions], " contributing contraction patterns"]];
      
      (* Sum woven_coeff * ComputeContractions[R, S, {pairs}] for each pair *)
      matrixElement = Total[
        KeyValueMap[
          #2 * ComputeContractions[R, S, {#1}] &,
          wovenContractions
        ]
      ];
      
      (* Normalize *)
      matrixElement = matrixElement / Sqrt[Normalization[R] * Normalization[S]];
      
      If[verbose,
        Print["------------------------------------"];
        Print["Matrix element: ", matrixElement];
        Print["------------------------------------"];
      ];
      
      matrixElement,
      
      (* === Original path: decompose and compute each overlap individually === *)
      collectedTerms = DecomposeTrace[ops, perm, 1, Verbose -> False, PrintTable -> False];
      
      If[verbose,
        Print["------------------------------------"];
        Print["Decomposition: ", collectedTerms];
        Print["------------------------------------"];
      ];
      
      (* Sum overlaps for fixed operator type *)
      sumByType[type_, permToCoeffAssoc_] := 
        Total[
          KeyValueMap[
            #2 * ComputeSingleOverlap[R, S, type, #1, Verbose -> verbose, Efficient -> efficient] &,
            permToCoeffAssoc
          ]
        ];
      
      (* Sum over all types *)
      matrixElement = Total[
        KeyValueMap[sumByType, collectedTerms]
      ];
      
      If[verbose,
        Print["------------------------------------"];
        Print["Matrix element: ", matrixElement];
        Print["------------------------------------"];
      ];
      
      matrixElement
    ]
  ];


(* ::Section::Closed:: *)
(*Brute-force computation*)


ComputeBruteForce::usage = 
  "ComputeContractionsBruteForce[R, S, K, dim] computes the overlap <R|tr(X^K)|S> using explicit operator \
construction and character theory. This 'brute-force' approach serves as a validation check.";

ComputeBruteForce[R_, S_, ops_String, perm_List, dim_, mass_:1] := 
  Module[
    {nR, nS, indexRepR, indexRepS, charTableR, charTableS, 
     conjClR, conjClS, sizeConjClR, sizeConjClS, 
     A, Ad, H, myFunction, myCoeff, result},
    
    (* ===== Step 1: Setup and initialization ===== *)
    
    (* Partition sizes *)
    nR = Total[R];
    nS = Total[S];
    
    (* Find representation indices in character tables *)
    indexRepR = GetRepresentationIndex[R];
    indexRepS = GetRepresentationIndex[S];
    
    (* Load character tables for S_nR and S_nS *)
    charTableR = LoadCharacterTable[nR]["CharacterTable"];
    charTableS = LoadCharacterTable[nS]["CharacterTable"];
    
    (* Get conjugacy classes (represented as partitions) *)
    conjClR = LoadCharacterTable[nR]["CharacterParameters"];
    conjClS = LoadCharacterTable[nS]["CharacterParameters"];
    
    (* Sizes of conjugacy classes *)
    sizeConjClR = LoadCharacterTable[nR]["SizesConjugacyClasses"];
    sizeConjClS = LoadCharacterTable[nS]["SizesConjugacyClasses"];
    
    (* Define ladder operators with indices *)
    A[i_, j_] := LadderAlgebra`a[i, j];
    Ad[i_, j_] := LadderAlgebra`ad[i, j];
    
    (* Define the Hamiltonian *)
    H := OperatorTesting`ExpandImplicitIndices[MonomialTrace[ops, perm, mass], dim]/.d->dim;
    
    (* ===== Step 2: Define computation for each conjugacy class pair ===== *)
    
    (* Compute vacuum expectation value for a pair of conjugacy class representatives *)
    myFunction[iR_, iS_] := 
      Module[{sigmaR, sigmaS, opR, opS, opRExtended, opSExtended, result},
        
        (* Create cycle representatives for conjugacy classes *)
        sigmaR = CreateCyclesFromPartition[conjClR[[iR]]];
        sigmaS = CreateCyclesFromPartition[conjClS[[iS]]];
        
        (* Build trace operators: Tr_sigma(a...a) and Tr_sigma(ad...ad) *)
        opR = LadderAlgebra`trSigma[sigmaR, Table[A, {i, nR}] /. List -> Sequence];
        opS = LadderAlgebra`trSigma[sigmaS, Table[Ad, {i, nS}] /. List -> Sequence];
        
        (* Expand implicit indices to explicit form *)
        opRExtended = OperatorTesting`ExpandImplicitIndices[opR, dim];
        opSExtended = OperatorTesting`ExpandImplicitIndices[opS, dim];
        
        (* Compute vacuum expectation value: <0|opR opS|0> *)
        result = LadderAlgebra`VacuumBra[dim, dim] ** opRExtended ** H ** opSExtended ** LadderAlgebra`VacuumKet[dim, dim];
        result
      ];
    
    (* ===== Step 3: Character theory weighting coefficient ===== *)
    
    (* Weight each conjugacy class contribution by character values and class sizes *)
    myCoeff[iR_, iS_] := 
      (charTableR[[indexRepR, iR]] * charTableS[[indexRepS, iS]] * 
       sizeConjClR[[iR]] * sizeConjClS[[iS]]) / 
      (Factorial[nR] * Factorial[nS]);
    
    (* ===== Step 4: Sum over all conjugacy class pairs ===== *)
    
    result = Sum[
      myCoeff[iR, iS] * myFunction[iR, iS],
      {iR, 1, Length[conjClR]},
      {iS, 1, Length[conjClS]}
    ];
    
    (* ===== Step 5: Normalize and substitute dimension ===== *)
    
    result / Sqrt[Normalization[R] * Normalization[S]] /. d -> dim
  ];


(* ::Section::Closed:: *)
(*Hamiltonian Matrix Construction*)


(* ::Subsection::Closed:: *)
(*Partition List*)


PartitionList[maxLambda_Integer] :=
  Flatten[Table[IntegerPartitions[lambda], {lambda, 0, maxLambda}], 1];


(* ::Subsection::Closed:: *)
(*Free Hamiltonian*)


Options[ComputeHfree] = {
  GroundStateOnly -> False
};

ComputeHfree[maxLambda_Integer, opts : OptionsPattern[]] :=
  Module[{groundStateOnly, RList, size},
    groundStateOnly = OptionValue[GroundStateOnly];
    
    RList = PartitionList[maxLambda];
    
    (* Filter to even excitations only if computing ground state *)
    If[groundStateOnly,
      RList = Select[RList, EvenQ[Total[#]] &]
    ];
    
    size = Length[RList];
    DiagonalMatrix[
      Table[d^2/2 + Total[RList[[i]]], {i, 1, size}]
    ]
  ];


(* ::Subsection::Closed:: *)
(*Interaction Hamiltonian*)


Options[ComputeHInt] = {
  Verbose -> False,
  PrecomputedContractions -> None,
  GroundStateOnly -> False
};

ComputeHInt[ops_String, perm_List, maxLambda_Integer, opts : OptionsPattern[]] :=
  Module[{verbose, precomputed, groundStateOnly, RList, size, wovenData, 
          contractionCache, getContraction, HInt},
    
    verbose = OptionValue[Verbose];
    precomputed = OptionValue[PrecomputedContractions];
    groundStateOnly = OptionValue[GroundStateOnly];
    
    (* Build the partition list *)
    RList = PartitionList[maxLambda];
    
    (* Filter to even excitations only if computing ground state *)
    If[groundStateOnly,
      RList = Select[RList, EvenQ[Total[#]] &]
    ];
    
    size = Length[RList];
    
    If[verbose, Print["Partition list: ", size, " partitions up to Lambda=", maxLambda]];
    
    (* Import woven contractions once *)
    wovenData = ImportWovenContractions[ops, perm, maxLambda];
    
    If[verbose, Print["Woven contractions loaded: ", Length[wovenData], " (nL,nR) pairs"]];
    
    (* Get or compute a contraction value, caching the result *)
    getContraction[pairs_, R_List, S_List] :=
      Module[{key, val},
        key = {pairs, R, S};
        If[AssociationQ[precomputed],
          If[KeyExistsQ[precomputed, key],
            precomputed[key],
            0],
            Simplify@ComputeContractions[R, S, {pairs}]
          ]
      ];
    
    (* Compute matrix elements *)
    HInt = Table[
      Module[{nL, nR, wc, me},
        nL = Total[RList[[iR]]];
        nR = Total[RList[[iS]]];
        
        If[nL>nR,
        0,
        
        (* Look up woven contractions for this excitation level pair *)
        wc = wovenData[{nL, nR}];
        
        If[MissingQ[wc],
          (* No contributing contractions *)
          0,
          
          (* Sum: woven_coeff * ComputeContractions[R, S, {pairs}] *)
          me = Total[
            KeyValueMap[
              #2 * getContraction[#1, RList[[iR]], RList[[iS]]] &,
              wc
            ]
          ];
          
          (* Normalize *)
          me / Sqrt[Normalization[RList[[iR]]] * Normalization[RList[[iS]]]]
        ]
        ]
      ],
      {iR, 1, size}, {iS, 1, size}
    ];
    
    (* Use symmetry to populate the entries below the diagonal *)
    Do[If[i>j, HInt[[i,j]]=HInt[[j,i]], Nothing], {i, 1, Dimensions[HInt][[1]]}, {j, 1, Dimensions[HInt][[2]]}];
    
    Simplify[HInt, Assumptions -> {d \[Element] Integers, d >= maxLambda}]
  ];


(* ::Section:: *)
(*Woven Contractions*)


(* ::Subsection::Closed:: *)
(*Helper: Decomposition by Type and Sigma*)


decompositionByTypeAndSigma[nL_Integer, nR_Integer, type_List, sigma_List, coeff_] :=
  Module[{check, canonicalizedExpr, pairsAndCoeffsAssociation},
    
    (* Check excitation balance: nR must equal nL + #a - #ad *)
    check = nR =!= nL + Count[type, LadderAlgebra`a] - Count[type, LadderAlgebra`ad];
    
    If[check,
      (* Excitations don't balance: return zero contribution *)
      Return[<|{} -> 0|>],
      
      (* Compute the overlap and extract contraction pairs *)
      canonicalizedExpr = ComputeOverlap[nL, type, sigma, Verbose -> False, Efficient -> True];
      pairsAndCoeffsAssociation = GetPairsAndCoefficientsAssociation[canonicalizedExpr];
      Return[coeff * # & /@ pairsAndCoeffsAssociation];
    ];
  ];
  
  
(* Helper function to print decomposition as a table *)
printWovenTable[collectedTerms_Association] := 
  Module[{rows, header},
    header = {"Sector", "Permutation", "Coefficient"};
    
    rows = Flatten[
      KeyValueMap[
        Function[{type, permToCoeff},
          MapIndexed[
            Function[{permCoeffPair, idx},
              If[idx[[1]] == 1,
                {type, permCoeffPair[[1]], permCoeffPair[[2]]},
                {Null, permCoeffPair[[1]], permCoeffPair[[2]]}
              ]
            ],
            Normal[permToCoeff]
          ]
        ],
        collectedTerms
      ],
      1
    ];
    
    Print[Grid[
      Prepend[rows, header],
      Frame -> All,
      Background -> {None, {1 -> LightBlue}},
      Alignment -> {Left, Center}
    ]];
  ];


(* ::Subsection::Closed:: *)
(*Compute Contributing Woven Contractions*)


ComputeContributingWovenContractions[nL_Integer, nR_Integer, collectedTerms_Association] :=
  Module[{myFunction, pairsAndCoeffsAssociationFlatten, collectedWovenContractions},
    
    (* Flatten the nested association type -> (perm -> coeff) into {type, perm, coeff} triples *)
    myFunction[type_, permToCoeffAssoc_] := KeyValueMap[{type, #1, #2} &, permToCoeffAssoc];
    
    (* Apply decomposition to each triple *)
    pairsAndCoeffsAssociationFlatten =
      decompositionByTypeAndSigma[nL, nR, #1, #2, #3] & @@@
        Flatten[KeyValueMap[myFunction, collectedTerms], 1];
    
    (* Merge all contributions with matching contraction patterns *)
    collectedWovenContractions = FullSimplify /@ Merge[pairsAndCoeffsAssociationFlatten, Total];
    
    (* Convert the residual Subscript[P, _] symbols to 1 *)
    collectedWovenContractions = collectedWovenContractions/.{Subscript[P, _] -> 1, TensorProduct -> Times};
    
    (* Remove keys with vanishing contribution *)
    Select[collectedWovenContractions, Simplify[#] =!= 0 &]
  ];


(* ::Subsection::Closed:: *)
(*Compute All Woven Contractions Up to Cutoff*)


Options[ComputeAllWovenContractions] = {
  Verbose -> False,
  PrintTable -> False
};

ComputeAllWovenContractions[ops_String, perm_List, Lambda_Integer, mass_, opts : OptionsPattern[]] :=
  Module[{verbose, printTable, collectedTerms, result, wc},
    
    verbose = OptionValue[Verbose];
    printTable = OptionValue[PrintTable];
    
    (* Decompose Tr_perm(M(ops)) once *)
    collectedTerms = DecomposeTrace[ops, perm, mass, Verbose -> False, PrintTable -> False];
    
    If[verbose, Print["Computing woven contractions for ops=", ops, ", perm=", perm, ", m=", mass,", Lambda=", Lambda, "..."]];
    
    (* Iterate over all excitation level pairs *)
    result = Association[];
    Do[
      wc = ComputeContributingWovenContractions[nL, nR, collectedTerms];
      If[Length[wc] > 0,
        result[{nL, nR}] = wc;
      ];

      , {nL, 0, Lambda}, {nR, 0, Lambda}
    ];
    
    If[verbose, Print["Done. ", Length[result], " non-trivial (nL,nR) pairs out of ", (Lambda+1)^2, "."]];
    
    (* Print table *)
    If[printTable,
      Print["Decomposition of Tr_perm(", ops, "):"];
      Print["=" <> StringRepeat["=", 50]];
      printWovenTable[result];
    ];
    
    result
  ];


(* ::Subsection::Closed:: *)
(*Export Woven Contractions to JSON*)


(* Convert contraction pairs to a 0-indexed permutation array *)
pairsToPermutation0Indexed[pairs_List] :=
  Module[{n, perm},
    n = Length[pairs];
    (* pairs is a list of {i, j} where i -> j; convert to 0-indexed *)
    perm = Table[0, {2 * n}];
    Do[
      perm[[pairs[[k, 1]]]] = pairs[[k, 2]] - 1,
      {k, 1, n}
    ];
    (* Complete the inverse mapping for the second block *)
    Do[
      perm[[pairs[[k, 2]]]] = pairs[[k, 1]] - 1,
      {k, 1, n}
    ];
    perm
  ];

(* Convert a polynomial coefficient in d to a list of [re_num, re_den, im_num, im_den] 4-tuples *)
(* Young projector symbols P_l, P_r and their TensorProduct are implicit *)
(* in every woven contraction, so we strip them before extracting coefficients *)
coeffToRationalList[coeff_] :=
  Module[{poly, coeffList},
    poly = Expand[coeff];
    coeffList = CoefficientList[poly, d];
    (* Convert each coefficient to [re_num, re_den, im_num, im_den] *)
    Map[
      Function[c,
        Module[{re, im},
          re = ComplexExpand[Re[c]] // Simplify;
          im = ComplexExpand[Im[c]] // Simplify;
          {Numerator[re], Denominator[re], Numerator[im], Denominator[im]}
        ]
      ],
      coeffList
    ]
  ];

(* Helper: build filename prefix from ops and perm *)
wovenFilePrefix[ops_String, perm_List] :=
  "wc_op_" <> ops <> "_p" <> StringJoin[ToString /@ perm];

ExportWovenContractions[ops_String, perm_List, mass_, wovenData_Association, outputDir_String] :=
  Module[{Lambda, prefix, existingFiles, existingLambdas, maxExisting,
          jsonEntries, jsonData, filename, fullPath, perm0},
    
    (* Determine Lambda from the keys *)
    Lambda = Max[Flatten[Keys[wovenData]]];
    
    (* Ensure output directory exists *)
    If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];
    
    (* Build filename prefix *)
    prefix = wovenFilePrefix[ops, perm];
    
    (* Check for existing files with same ops, perm, mass *)
    existingFiles = FileNames[prefix <> "_m" <> ToString[N[mass]] <> "_Lambda*.json", outputDir];
    If[Length[existingFiles] > 0,
      existingLambdas = ToExpression /@ 
        StringCases[existingFiles, "Lambda" ~~ x : DigitCharacter.. ~~ ".json" :> x];
      existingLambdas = Flatten[existingLambdas];
      maxExisting = Max[existingLambdas];
      
      If[maxExisting >= Lambda,
        Echo[StringJoin[{"File with ops=", ops, " perm=", ToString[perm], " m=", ToString[mass], " Lambda=", ToString[maxExisting], " >= ", ToString[Lambda], " already exists. Skipping export."}]];
        Return[$Failed]
      ];
      
      (* Delete smaller Lambda files *)
      Do[
        Echo[StringJoin[{"Removing outdated file: ", ToString[f]}]];
        DeleteFile[f],
        {f, existingFiles}
      ];
    ];
    
    (* Convert 1-indexed perm to 0-indexed for JSON *)
    perm0 = perm - 1;
    
    (* Build JSON-compatible structure *)
    jsonEntries = KeyValueMap[
      Function[{nlnr, contractions},
        Module[{key, entries},
          key = ToString[nlnr[[1]]] <> "_" <> ToString[nlnr[[2]]];
          entries = KeyValueMap[
            Function[{pairs, coeff},
              <|
                "permutation" -> pairsToPermutation0Indexed[pairs],
                "coefficient_poly_d" -> coeffToRationalList[coeff]
              |>
            ],
            contractions
          ];
          key -> entries
        ]
      ],
      wovenData
    ];
    
    jsonData = <|
      "operators" -> ops,
      "trace_permutation" -> perm0,
      "Lambda" -> Lambda,
      "mass" -> {Numerator[mass], Denominator[mass]},
      "is_even" -> EvenMonomialQ[ops],
      "is_hermitian" -> HermitianMonomialQ[ops, perm, mass],
      "description" -> StringJoin["Woven contractions C*tr(tau*(P_R x P_S)) contributing to <R|Tr_perm(", ops, ")|S>. ",
        "Permutations are 0-indexed. Coefficients are polynomials in d stored as lists of ",
        "[re_num, re_den, im_num, im_den] rational 4-tuples, from degree 0 upward."],
      "woven_contractions" -> Association[jsonEntries]
    |>;
    
    (* Export *)
    filename = StringJoin[prefix, "_m", ToString[N[mass]], "_Lambda", ToString[Lambda], ".json"];
    fullPath = FileNameJoin[{outputDir, filename}];
    Export[fullPath, jsonData, "JSON"];
    
    Echo[StringJoin[{"Exported woven contractions to: ", fullPath}]];
    fullPath
  ];

(* Convenience overload: use default output directory *)
ExportWovenContractions[ops_String, perm_List, wovenData_Association] :=
  ExportWovenContractions[ops, perm, 1/2, wovenData,
    FileNameJoin[{$PackageRoot, "data", "processed", "woven_contractions"}]
  ];


(* ::Subsection::Closed:: *)
(*Import Woven Contractions from JSON*)


(* Convert a 0-indexed permutation array back to contraction pairs *)
permutation0IndexedToPairs[perm_List] :=
  Module[{n},
    n = Length[perm] / 2;
    (* Reconstruct pairs from the first n entries: i -> perm[i], both shifted to 1-indexed *)
    Table[{i, perm[[i]] + 1}, {i, 1, n}]
  ];

(* Convert a list of rational 4-tuples [re_num, re_den, im_num, im_den] back to a polynomial in d *)
rationalListToCoeff[rationalQuads_List] :=
  Module[{coeffs},
    coeffs = Map[
      Function[q,
        If[Length[q] == 4,
          (* 4-tuple: complex coefficient *)
          q[[1]] / q[[2]] + I * q[[3]] / q[[4]],
          (* 2-tuple: legacy real coefficient *)
          q[[1]] / q[[2]]
        ]
      ],
      rationalQuads
    ];
    Sum[coeffs[[i]] * d^(i - 1), {i, 1, Length[coeffs]}]
  ];

ImportWovenContractions[ops_String, perm_List, Lambda_Integer, inputDir_String] :=
  Module[{prefix, existingFiles, existingLambdas, bestLambda,
          filename, fullPath, jsonData, wcData, result, filtered},
    
    (* Build filename prefix *)
    prefix = wovenFilePrefix[ops, perm];
    
    (* Check for existing files with same ops and perm *)
    existingFiles = FileNames[prefix <> "_m*_Lambda*.json", inputDir];
    
    If[Length[existingFiles] == 0,
      (* No files exist, compute and export *)
      Echo[StringJoin[{"No file found for ops=", ops, " perm=", ToString[perm], ". Computing woven contractions for Lambda=", ToString[Lambda], "..."}]];
      result = ComputeAllWovenContractions[ops, perm, Lambda, 1/2];
      ExportWovenContractions[ops, perm, wovenData, inputDir];
      Return[result]
    ];
    
    (* Extract Lambda values from existing files *)
    existingLambdas = ToExpression /@ 
      StringCases[existingFiles, "Lambda" ~~ x : DigitCharacter.. ~~ ".json" :> x];
    existingLambdas = Flatten[existingLambdas];
    
    (* Find the best Lambda to use *)
    bestLambda = SelectFirst[Sort[existingLambdas, Greater], # >= Lambda &, None];
    
    If[bestLambda === None,
      (* All existing files have smaller Lambda, compute new data *)
      Print["Existing files have Lambda < ", Lambda, ". Computing woven contractions for Lambda=", Lambda, "..."];
      result = ComputeAllWovenContractions[ops, perm, Lambda, 1/2];
      ExportWovenContractions[ops, perm, result, inputDir];
      Return[result]
    ];
    
    (* Find the file with bestLambda *)
    fullPath = SelectFirst[existingFiles, StringContainsQ[#, "Lambda" <> ToString[bestLambda] <> ".json"] &];
    
    If[bestLambda > Lambda,
      Echo[StringJoin[{"Using file with Lambda=", ToString@bestLambda, " and filtering to Lambda=", ToString@Lambda}]]
    ];
    
    (* Import JSON *)
    jsonData = Import[fullPath, "RawJSON"];
    wcData = jsonData["woven_contractions"];
    
    (* Reconstruct the nested association *)
    result = Association[
      KeyValueMap[
        Function[{key, entries},
          Module[{nlnr, contractions},
            (* Parse "nL_nR" key back to {nL, nR} *)
            nlnr = ToExpression /@ StringSplit[key, "_"];
            
            (* Reconstruct the inner association: pairs -> coeff *)
            contractions = Association[
              Function[entry,
                permutation0IndexedToPairs[entry["permutation"]] -> 
                  rationalListToCoeff[entry["coefficient_poly_d"]]
              ] /@ entries
            ];
            
            nlnr -> contractions
          ]
        ],
        wcData
      ]
    ];
    
    (* Filter entries where any key component > Lambda *)
    If[bestLambda > Lambda,
      filtered = KeySelect[result, Max[#] <= Lambda &];
      filtered,
      result
    ]
  ];

(* Convenience overload: use default input directory *)
ImportWovenContractions[ops_String, perm_List, Lambda_Integer] :=
  ImportWovenContractions[ops, perm, Lambda,
    FileNameJoin[{$PackageRoot, "data", "processed", "woven_contractions"}]
  ];


(* ::Section:: *)
(*Protect Symbols*)


Protect[MonomialTrace, CyclicProduct, EvenMonomialQ, HermitianMonomialQ,
        DecomposeTrace, ReconstructTrace,
        ContractIndices, IdentifyTerms, CollectTermsByType,
        DeclareYoungProjector, ApplyWickTheorem, RelabelIndices,
        ConvertToTensorContractions, CanonicalizeTensorExpression, ComputeOverlap,
        PermutationTensorProduct, GetRepresentationIndex, ContractPermutationTensor,
        GetPairsAndCoefficientsAssociation, ComputeContractions, ComputeSingleOverlap,
        ComputeMatrixElementInteraction, ComputeBruteForce,
        ComputeHInt, ComputeHfree, PartitionList,
        ComputeContributingWovenContractions, ComputeAllWovenContractions, ExportWovenContractions, ImportWovenContractions,
        d, P];


End[];
EndPackage[];
