Read("main.g");

# Set the seed for reproducibility
Reset(GlobalMersenneTwister, 42);

# Set verbose level to 0 to suppress output
VerboseLevel := 3;
info := NewInfoClass("Info");
SetInfoLevel(info, VerboseLevel);

# Inputs
n:=4;
k:=0;
p:=n-2-k;
q:=n-2+k;

# Compute representatives
Print("-------------\n");
Print("Compute representatives and sizes for doubles cosets S_p g S_q, for g in S_n\n");
Print("-------------\n");
Print("Arguments (p,n,q) = ", [p,n,q], "\n\n");
t_before:=Runtime();
dc_reps_and_size:=GetDoubleCosetsRepsAndSizes(p, n, q);
t_after:=Runtime();
PrintInfo(info, 1, "Representatives and sizes", dc_reps_and_size);
Print("Number of double cosets: ", Size(dc_reps_and_size), "\n");
Print("CPU time spent with GAP functions (without child processes): ", t_after - t_before, " ms");

quit;