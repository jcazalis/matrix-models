LoadPackage( "ctbllib" );

#####################################
#########  HELPER FUNCTIONS #########
#####################################

# Set verbose level to 0 to suppress output
VerboseLevel := 0;
info := NewInfoClass("Info");
SetInfoLevel(info, VerboseLevel);

####################
#
# PrintInfo(info, level, name, obj)
#
# Prints the information about the object obj at the given level.
#
# Parameters
# ----------
# 	info: InfoClass
# 		The info class to use for printing.
# 	level: Integer
# 		The verbosity level for the info message.
# 	name: String
# 		A name or header for the information being printed.
# 	obj: Object
# 		The object whose parts will be printed.
#
# Returns
# -------
# 	nothing
#
PrintInfo := function(info, level, name, obj)
    local part;

    Info(info, level, name);
    Info(info, level, "-------------");
    for part in obj do
        Info(info, level, part);
    od;
    Info(info, level, "\n");
end;

####################
#
# ToMathematicaString(obj)
#
# Converts a GAP object's string representation to a Mathematica-compatible format
# by replacing square brackets with curly braces.
#
# Parameters
# ----------
# 	obj: Object
# 		The GAP object to convert.
#
# Returns
# -------
# 	String
# 		A string formatted for Mathematica.
#
ToMathematicaString := function( obj )
    local str;
    
    # Convert the GAP object to its string representation
    str := String(obj);
    
    # Replace standard list brackets with Mathematica curly braces
    str := ReplacedString(str, "[", "{");
    str := ReplacedString(str, "]", "}");
    
    return str;
end;


#####################################
##########  Double Cosets ###########
#####################################

####################
#
# GetDoubleCosetsRepsAndSizes(p, n, q)
#
# Returns the representatives for each double cosets of the symmetric group S_n
# quotient by its subgroups S_p (left) and by S_q (right). We use the canonical embedding, meaning S_p and S_q act
# on the indices [1..p] and [1..q] respectively. 
#
# Parameters
# ----------
# 	p: Integer
# 		Degree p for the left subgroup S_p
# 	n: Integer
# 		Degree n for the supergroup S_n, must be larger than max(p,q)
# 	q: Integer
# 		Degree q for the right subgroup S_q
#
# Returns
# -------
# 	List of Tuples
# 		A list of tuples, each containing a representative of a double coset and its size.
#
GetDoubleCosetsRepsAndSizes := function(p, n, q)
    local G, Hp, Hq, my_string1, my_string2;

    if Maximum(p,q)>n then
        my_string1 := "Invalid argument for GetDoubleCosetsRepsAndSizes(p, n, q): (p,n,q)";
        my_string2 := JoinStringsWithSeparator([my_string1, String([p, n, q])], " = ");
        Info(info, 2, my_string2);
        return fail;
    else
        G := SymmetricGroup(n);
        Hp := SymmetricGroup(p);
        Hq := SymmetricGroup(q);
        return DoubleCosetRepsAndSizes(G, Hp, Hq);
    fi;
end;

StoreDoubleCosetsRepsAndSizes := function(p, n, q, filename)
    local dc_reps_and_size, reps, sizes, f;

    dc_reps_and_size := GetDoubleCosetsRepsAndSizes(p, n, q);
    reps := List(dc_reps_and_size, x->ListPerm(x[1], n));
    sizes := List(dc_reps_and_size, x->x[2]);

    f := OutputTextFile(filename, false);
    SetPrintFormattingStatus(f, false);

    PrintTo(f, "DoubleCosetsRepsAndSizes[", p, ",", n, ",", q, "] =\n");
    PrintTo(f, "  {\"Representatives\"->");
    PrintTo(f, ToMathematicaString(reps), ",\n");

    PrintTo(f, "   \"Sizes\"->");
    PrintTo(f, "{", JoinStringsWithSeparator(sizes), "}}\n");

    CloseStream(f);

end;

StoreDoubleCosetsRepsAndSizesBatch := function(arg)
    # max_offset controls the range of (p,q) pairs relative to n.
    # even mode: max_offset = K/2
    # odd mode:  max_offset = (K-1)/2
    # odd_difference=false -> p-q is even (legacy behavior)
    # odd_difference=true  -> p-q is odd
    local n1, n2, max_offset, odd_difference, n, p, q, offset, k, filename, base_dir;

    if Length(arg) = 3 then
        n1 := arg[1];
        n2 := arg[2];
        max_offset := arg[3];
        odd_difference := false;
    elif Length(arg) = 4 then
        n1 := arg[1];
        n2 := arg[2];
        max_offset := arg[3];
        odd_difference := arg[4];
    else
        Error("StoreDoubleCosetsRepsAndSizesBatch expects 3 or 4 arguments: (n1, n2, max_offset[, odd_difference])");
    fi;

    if IsBoundGlobal("OUTPUT_DIR") then
        base_dir := Concatenation(ValueGlobal("OUTPUT_DIR"), "/");
    else
        base_dir := "../processed/double_cosets/";
    fi;
    for n in [n1..n2] do
        for offset in [0..max_offset] do
            if odd_difference then
                for k in [-offset..offset+1] do
                    p := n - offset - k;
                    q := n - offset - 1 + k;
                    if Minimum(p, q) >= 0 then
                        filename := Concatenation(base_dir, "dc_n", String(n), "_p", String(p), "_q", String(q), ".txt");
                        if IsExistingFile(filename) then
                            Print("File ", filename, " already exists. Pass.\n");
                        else
                            StoreDoubleCosetsRepsAndSizes(p, n, q, filename);
                            Print("File ", filename, " created.\n");
                        fi;
                    fi;
                od;
            else
                for k in [-offset..offset] do
                    p := n - offset + k;
                    q := n - offset - k;
                    if Minimum(p, q) >= 0 then
                        filename := Concatenation(base_dir, "dc_n", String(n), "_p", String(p), "_q", String(q), ".txt");
                        if IsExistingFile(filename) then
                            Print("File ", filename, " already exists. Pass.\n");
                        else
                            StoreDoubleCosetsRepsAndSizes(p, n, q, filename);
                            Print("File ", filename, " created.\n");
                        fi;
                    fi;
                od;
            fi;
        od;
    od;
end;


#####################################
#########  Conjugacy classes ########
#####################################

####################
#
# StoreSymmetricCharacterTable(n, compute_elements, filename)
#
# Computes and stores the character table of the symmetric group S_n into a
# Mathematica-formatted text file.
#
# Parameters
# ----------
# 	n: Integer
# 		The degree of the symmetric group S_n.
# 	compute_elements: Boolean
# 		If true, also computes and stores all elements of each conjugacy class.
# 	filename: String
# 		The name of the output file.
#
# Returns
# -------
# 	nothing
#
StoreSymmetricCharacterTable := function(n, compute_elements, filename)
    local G, c, ct, ccl, elements, f;

    if n=0 then

        f := OutputTextFile(filename, false);
        SetPrintFormattingStatus(f, false);
  
        PrintTo(f, "SymmetricCharacterTable[0] =\n");
        PrintTo(f, "  {\"CharacterParameters\"->{{}},\n");
        PrintTo(f, "   \"SizesConjugacyClasses\"->{1},\n");
        PrintTo(f, "   \"ElementsConjugacyClasses\"->{ { {  } } },\n");
        PrintTo(f, "   \"CharacterTable\"->{{1}}}\n");
    
        CloseStream(f);

    else

        G := SymmetricGroup(n);
        c := CharacterTable("Symmetric", n);
        ct := CharacterTable(G);
        ccl := ConjugacyClasses(ct);

        if compute_elements then 
            elements := List(ccl, x-> List(Elements(x), y->ListPerm(y,n))); 
        fi;

        f := OutputTextFile(filename, false);
        SetPrintFormattingStatus(f, false);

        PrintTo(f, "SymmetricCharacterTable[", n, "] =\n");
        PrintTo(f, "  {\"CharacterParameters\"->");
        PrintTo(f, "{", JoinStringsWithSeparator(List(CharacterParameters(c), x->Concatenation("{",JoinStringsWithSeparator(x[2]),"}"))), "},\n");

        PrintTo(f, "   \"SizesConjugacyClasses\"->");
        PrintTo(f, "{", JoinStringsWithSeparator(List(SizesConjugacyClasses(ct), x->x)), "},\n");

        if compute_elements then
            PrintTo(f, "   \"ElementsConjugacyClasses\"->");
            PrintTo(f, ToMathematicaString(elements), ",\n");
        fi;

        PrintTo(f, "   \"CharacterTable\"->");
        PrintTo(f, "{", JoinStringsWithSeparator(List(Irr(ct), x->Concatenation("{",JoinStringsWithSeparator(x),"}"))), "}}\n");

        CloseStream(f);

    fi;
end;;

####################
#
# StoreSymmetricCharacterTableBatch(n1, n2)
#
# A batch function to store character tables for a range of symmetric groups S_n.
#
# Parameters
# ----------
# 	n1: Integer
# 		The starting degree for the symmetric groups.
# 	n2: Integer
# 		The ending degree for the symmetric groups.
#
# Returns
# -------
# 	nothing
#
StoreSymmetricCharacterTableBatch := function(n1, n2)
    local n, filename;
    for n in [n1..n2] do
        if IsBoundGlobal("OUTPUT_DIR") then
            filename := Concatenation(ValueGlobal("OUTPUT_DIR"), "/ssct_", String(n), ".txt");
        else
            filename := JoinStringsWithSeparator(["../processed/conjugacy_classes/ssct_", String(n), ".txt"],"");
        fi;
        if IsExistingFile(filename) then
            Print("File ", filename, " already exists. Pass.\n");
        else
            StoreSymmetricCharacterTable(n, true, filename);
            Print("File ", filename, " created.\n");
       fi;
    od;
end;


#####################################
####  Conjugacy classes (Python) ####
#####################################

####################
#
# StoreSymmetricCharacterTablePython(n, filename)
#
# Computes and stores the character table of the symmetric group S_n into a
# JSON-formatted text file for Python import.
#
# Parameters
# ----------
# 	n: Integer
# 		The degree of the symmetric group S_n.
# 	filename: String
# 		The name of the output file.
#
# Returns
# -------
# 	nothing
#
StoreSymmetricCharacterTablePython := function(n, filename)
    local G, c, ct, ccl, f;

    if n=0 then

        f := OutputTextFile(filename, false);
        SetPrintFormattingStatus(f, false);
  
        PrintTo(f, "{\n");
        PrintTo(f, "  \"n\": 0,\n");
        PrintTo(f, "  \"CharacterParameters\": [[]],\n");
        PrintTo(f, "  \"SizesConjugacyClasses\": [1],\n");
        PrintTo(f, "  \"CharacterTable\": [[1]]\n");
        PrintTo(f, "}\n");
    
        CloseStream(f);

    else

        G := SymmetricGroup(n);
        c := CharacterTable("Symmetric", n);
        ct := CharacterTable(G);
        ccl := ConjugacyClasses(ct);

        f := OutputTextFile(filename, false);
        SetPrintFormattingStatus(f, false);

        PrintTo(f, "{\n");
        PrintTo(f, "  \"n\": ", n, ",\n");
        PrintTo(f, "  \"CharacterParameters\": ");
        PrintTo(f, "[", JoinStringsWithSeparator(List(CharacterParameters(c), x->Concatenation("[",JoinStringsWithSeparator(x[2]),"]"))), "],\n");

        PrintTo(f, "  \"SizesConjugacyClasses\": ");
        PrintTo(f, "[", JoinStringsWithSeparator(List(SizesConjugacyClasses(ct), x->x)), "],\n");

        PrintTo(f, "  \"CharacterTable\": ");
        PrintTo(f, "[", JoinStringsWithSeparator(List(Irr(ct), x->Concatenation("[",JoinStringsWithSeparator(x),"]"))), "]\n");
        
        PrintTo(f, "}\n");

        CloseStream(f);

    fi;
end;;

####################
#
# StoreSymmetricCharacterTablePythonBatch(n1, n2)
#
# A batch function to store character tables for a range of symmetric groups
# S_n, into a JSON-formatted text file for Python import.
#
# Parameters
# ----------
# 	n1: Integer
# 		The starting degree for the symmetric groups.
# 	n2: Integer
# 		The ending degree for the symmetric groups.
#
# Returns
# -------
# 	nothing
#
StoreSymmetricCharacterTablePythonBatch := function(n1, n2)
    local n, filename;
    for n in [n1..n2] do
        if IsBoundGlobal("OUTPUT_DIR") then
            filename := Concatenation(ValueGlobal("OUTPUT_DIR"), "/ct_", String(n), ".json");
        else
            filename := JoinStringsWithSeparator(["../processed/character_tables/ct_", String(n), ".json"],"");
        fi;
        if IsExistingFile(filename) then
            Print("File ", filename, " already exists. Pass.\n");
        else
            StoreSymmetricCharacterTablePython(n, filename);
            Print("File ", filename, " created.\n");
       fi;
    od;
end;