Read("main.g");

# Set the seed for reproducibility
Reset(GlobalMersenneTwister, 42);

# Set verbose level to 0 to suppress output
VerboseLevel := 1;
info := NewInfoClass("Info");
SetInfoLevel(info, VerboseLevel);

# Inputs 
# nmax:=9; # Maximum order n for symmetric group S_n

# Compute representatives
Print("-------------\n");
Print("Compute the character table and elements of conjugacy classes of S_n for n up to ", nmax, "\n");
Print("-------------\n");
t_before:=Runtime();
StoreSymmetricCharacterTableBatch(0, nmax);
t_after:=Runtime();
Print("CPU time spent with GAP functions (without child processes): ", t_after - t_before, " ms");

quit;