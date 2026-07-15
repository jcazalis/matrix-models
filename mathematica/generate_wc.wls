#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Import package *)
PrependTo[$Path,FileNameJoin[{Directory[],"packages"}]];
Needs["LadderAlgebra`"];
Needs["OperatorTesting`"];
Needs["MatrixModels`"];

(* Defaults *)
defaultOps="XXXX";
defaultPerm="2341";
defaultLambda=14;
defaultMass=1/2;
defaultOutputDir=FileNameJoin[{ParentDirectory[], "data", "processed", "woven_contractions"}];

(* Parse CLI arguments: wolframscript -script generate_wc.wls [ops] [perm] [Lambda] [mass] [outputDir]*)
args=Rest[$ScriptCommandLine];
ops=If[Length[args]>=1, args[[1]], defaultOps];
permStr=If[Length[args]>=2, args[[2]], defaultPerm];
Lambda=If[Length[args]>=3, ToExpression[args[[3]]], defaultLambda];
mass=If[Length[args]>=4, ToExpression[args[[4]]], defaultMass];
outputDir=If[Length[args]>=5, args[[5]], defaultOutputDir];

(* Parse perm string "2341" -> {2, 3, 4, 1} (1-indexed) *)
perm=ToExpression[Characters[permStr]];

(* Code to execute *)
wovenData=ComputeAllWovenContractions[ops,perm,Lambda,mass,Verbose->True];
ExportWovenContractions[ops,perm,mass,wovenData,outputDir]
