#####################################
####  Coset Representatives  #######
#####################################

# Computes right coset representatives of S_n / H_n (and S_m / H_m)
# where H_n is the projection of the centralizer of tau in S_n x S_m.
#
# Uses GAP's Centralizer, ActionHomomorphism, and RightTransversal.
#
# Input:  a GAP source file with a list of rec(tau_1indexed, n, m) records.
# Output: a JSON file with coset representatives (0-indexed).
#
# Usage:
#   1. Python generates a GAP input file (e.g., _coset_input.g) containing:
#        coset_entries := [ rec(tau_1indexed := [...], n := ..., m := ...), ... ];
#   2. A driver script (generate_coset_reps.g) reads this file, then calls:
#        ProcessCosetBatch(coset_entries, "output.json");
#

# ============================================================
# Core function: compute coset data for a single (tau, n, m)
# ============================================================

####################
#
# ComputeCosetReps(tau_list_1indexed, n, m)
#
# Given a permutation tau in S_{n+m} (as a 1-indexed list), computes the
# projection of the centralizer of tau in S_n x S_m onto each factor,
# and returns right coset representatives.
#
# The subgroup S_n x S_m is the setwise stabilizer of {1,...,n} in S_{n+m}.
# The centralizer C = {(sigma, nu) in S_n x S_m : (sigma x nu) tau = tau (sigma x nu)}.
# H_n = projection of C onto S_n (restriction to {1,...,n}).
# H_m = projection of C onto S_m (restriction to {n+1,...,n+m}).
#
# Parameters
# ----------
#   tau_list_1indexed: List
#       The permutation tau as a 1-indexed list of length n+m.
#   n: Integer
#       Size of the first block (S_n).
#   m: Integer
#       Size of the second block (S_m).
#
# Returns
# -------
#   Record with fields:
#     .left_h_order     : order of H_n
#     .left_num_reps    : |S_n / H_n|
#     .left_reps        : list of coset reps (1-indexed lists of length n)
#     .right_h_order    : order of H_m
#     .right_num_reps   : |S_m / H_m|
#     .right_reps       : list of coset reps (1-indexed lists of length m)
#
ComputeCosetReps := function(tau_list_1indexed, n, m)
    local nm, tau_perm, G_nm, SnxSm, C,
          phi_n, H_n, S_n, trans_n, reps_n,
          phi_m, H_m, S_m, trans_m, reps_m;

    nm := n + m;

    # Handle trivial cases
    if nm = 0 then
        return rec(
            left_h_order := 1, left_num_reps := 1,
            left_reps := [[]],
            right_h_order := 1, right_num_reps := 1,
            right_reps := [[]]
        );
    fi;

    # Build tau as a GAP permutation
    tau_perm := PermList(tau_list_1indexed);

    # S_n x S_m as the setwise stabilizer of {1,...,n} in S_{n+m}
    G_nm := SymmetricGroup(nm);
    if n = 0 or m = 0 then
        # One factor is trivial
        SnxSm := G_nm;
    else
        SnxSm := Stabilizer(G_nm, [1..n], OnSets);
    fi;

    # Centralizer of tau in S_n x S_m
    C := Centralizer(SnxSm, tau_perm);

    # --- Left side (S_n) ---
    if n = 0 then
        reps_n := [[]];
        H_n := Group(());
    else
        # Project C onto the action on {1,...,n}
        phi_n := ActionHomomorphism(C, [1..n]);
        H_n := Image(phi_n);
        S_n := SymmetricGroup(n);
        trans_n := RightTransversal(S_n, H_n);
        reps_n := List(trans_n, x -> ListPerm(x, n));
    fi;

    # --- Right side (S_m) ---
    if m = 0 then
        reps_m := [[]];
        H_m := Group(());
    else
        # Project C onto the action on {n+1,...,n+m}
        # ActionHomomorphism maps to S_m on {1,...,m}
        phi_m := ActionHomomorphism(C, [n+1..n+m]);
        H_m := Image(phi_m);
        S_m := SymmetricGroup(m);
        trans_m := RightTransversal(S_m, H_m);
        reps_m := List(trans_m, x -> ListPerm(x, m));
    fi;

    return rec(
        left_h_order := Size(H_n),
        left_num_reps := Length(reps_n),
        left_reps := reps_n,
        right_h_order := Size(H_m),
        right_num_reps := Length(reps_m),
        right_reps := reps_m
    );
end;


# ============================================================
# Single entry: compute and return as a record (for testing)
# ============================================================

####################
#
# PrintCosetInfo(tau_list_1indexed, n, m)
#
# Prints a human-readable summary of the coset reduction for one (tau, n, m).
#
PrintCosetInfo := function(tau_list_1indexed, n, m)
    local result;
    result := ComputeCosetReps(tau_list_1indexed, n, m);
    Print("tau = ", tau_list_1indexed, ", n = ", n, ", m = ", m, "\n");
    Print("  Left:  |H_n| = ", result.left_h_order,
          ", |S_n/H_n| = ", result.left_num_reps, "\n");
    Print("  Right: |H_m| = ", result.right_h_order,
          ", |S_m/H_m| = ", result.right_num_reps, "\n");
end;


# ============================================================
# Batch processing: read GAP input file, write output JSON
# ============================================================

####################
#
# WriteCosetOutput(entries, results, output_filename)
#
# Writes the computed coset data to a JSON file.
#
# Parameters
# ----------
#   entries: List of records
#       The input entries (each with tau_1indexed, n, m).
#   results: List of records
#       The computed coset data (from ComputeCosetReps).
#   output_filename: String
#       Path to the output JSON file.
#
WriteCosetOutput := function(entries, results, output_filename)
    local f, i, entry, result, tau_0indexed, rep, j, first_entry;

    f := OutputTextFile(output_filename, false);
    SetPrintFormattingStatus(f, false);

    PrintTo(f, "{\n");
    PrintTo(f, "  \"entries\": [\n");

    first_entry := true;
    for i in [1..Length(entries)] do
        entry := entries[i];
        result := results[i];

        # Convert tau from 1-indexed to 0-indexed
        tau_0indexed := List(entry.tau_1indexed, x -> x - 1);

        if not first_entry then
            PrintTo(f, ",\n");
        fi;
        first_entry := false;

        PrintTo(f, "    {\n");
        PrintTo(f, "      \"tau_0indexed\": [",
                JoinStringsWithSeparator(List(tau_0indexed, String), ", "),
                "],\n");
        PrintTo(f, "      \"n\": ", entry.n, ",\n");
        PrintTo(f, "      \"m\": ", entry.m, ",\n");

        # Left side
        PrintTo(f, "      \"left\": {\n");
        PrintTo(f, "        \"h_order\": ", result.left_h_order, ",\n");
        PrintTo(f, "        \"num_reps\": ", result.left_num_reps, ",\n");
        PrintTo(f, "        \"reps_0indexed\": [\n");
        for j in [1..result.left_num_reps] do
            rep := List(result.left_reps[j], x -> x - 1);
            PrintTo(f, "          [",
                    JoinStringsWithSeparator(List(rep, String), ", "),
                    "]");
            if j < result.left_num_reps then
                PrintTo(f, ",");
            fi;
            PrintTo(f, "\n");
        od;
        PrintTo(f, "        ]\n");
        PrintTo(f, "      },\n");

        # Right side
        PrintTo(f, "      \"right\": {\n");
        PrintTo(f, "        \"h_order\": ", result.right_h_order, ",\n");
        PrintTo(f, "        \"num_reps\": ", result.right_num_reps, ",\n");
        PrintTo(f, "        \"reps_0indexed\": [\n");
        for j in [1..result.right_num_reps] do
            rep := List(result.right_reps[j], x -> x - 1);
            PrintTo(f, "          [",
                    JoinStringsWithSeparator(List(rep, String), ", "),
                    "]");
            if j < result.right_num_reps then
                PrintTo(f, ",");
            fi;
            PrintTo(f, "\n");
        od;
        PrintTo(f, "        ]\n");
        PrintTo(f, "      }\n");

        PrintTo(f, "    }");
    od;

    PrintTo(f, "\n  ]\n");
    PrintTo(f, "}\n");

    CloseStream(f);
end;


####################
#
# ProcessCosetBatch(entries, output_filename)
#
# Computes coset representatives for each entry in the list and writes
# the results to a JSON output file.
#
# The entries list should be defined in a separate GAP file (generated by
# Python's prepare_coset_input.py) and Read() before calling this function.
# Each entry is a record with fields: tau_1indexed, n, m.
#
# Parameters
# ----------
#   entries: List of records
#       Each record has fields: tau_1indexed (list), n (int), m (int).
#   output_filename: String
#       Path to the output JSON file.
#
ProcessCosetBatch := function(entries, output_filename)
    local results, i, entry, result, t_before, t_after;

    Print("Processing ", Length(entries), " entries.\n");

    results := [];
    t_before := Runtime();

    for i in [1..Length(entries)] do
        entry := entries[i];
        if i mod 10 = 0 or i = Length(entries) then
            Print("  Processing entry ", i, "/", Length(entries),
                  " (n=", entry.n, ", m=", entry.m, ")\n");
        fi;
        result := ComputeCosetReps(entry.tau_1indexed, entry.n, entry.m);
        Add(results, result);
    od;

    t_after := Runtime();
    Print("Computation time: ", t_after - t_before, " ms\n");

    Print("Writing output to ", output_filename, "\n");
    WriteCosetOutput(entries, results, output_filename);
    Print("Done.\n");
end;
