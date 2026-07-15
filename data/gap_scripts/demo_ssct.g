Read("main.g");

# Set the seed for reproducibility
Reset(GlobalMersenneTwister, 42);

# Set verbose level to 0 to suppress output
VerboseLevel := 3;
info := NewInfoClass("Info");
SetInfoLevel(info, VerboseLevel);

# Inputs
n:=15;

# Compute representatives
Print("-------------\n");
Print("Compute and print the character table for S_n\n");
Print("-------------\n");
Print("Arguments n = ", n, "\n\n");
t_before:=Runtime();
StoreSymmetricCharacterTable(n, false, "log.txt");
t_after:=Runtime();
Print("CPU time spent with GAP functions (without child processes): ", t_after - t_before, " ms");

quit;