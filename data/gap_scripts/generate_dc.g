Read("main.g");

# Set the seed for reproducibility
Reset(GlobalMersenneTwister, 42);

# Set verbose level to 0 to suppress output
VerboseLevel := 1;
info := NewInfoClass("Info");
SetInfoLevel(info, VerboseLevel);

# Inputs 
# nmax:=15;      # Maximum order n for symmetric group S_n
# max_offset:=2; # Maximum offset for (p,q) pairs: K/2 (even K) or (K-1)/2 (odd K)
#                # Defaults to 2 (K=4) if not pre-defined by caller.
# odd_difference:=false; # If true, enforce odd p-q differences.
#                        # Defaults to false (legacy even-difference behavior).

# Default max_offset to 2 (equivalent to K=4) if not set by caller
if not IsBoundGlobal("max_offset") then
    max_offset := 2;
fi;

# Default odd_difference to false if not set by caller
if not IsBoundGlobal("odd_difference") then
    odd_difference := false;
fi;

# Compute representatives
Print("-------------\n");
Print("Compute representatives and sizes for doubles cosets S_p g S_q, for g in S_n for n up to ", nmax, ", max_offset = ", max_offset, ", odd_difference = ", odd_difference, "\n");
Print("-------------\n");
t_before:=Runtime();
StoreDoubleCosetsRepsAndSizesBatch(0, nmax, max_offset, odd_difference);
t_after:=Runtime();
Print("CPU time spent with GAP functions (without child processes): ", t_after - t_before, " ms");

quit;