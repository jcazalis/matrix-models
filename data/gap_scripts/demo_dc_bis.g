Read("main.g");

# Set the seed for reproducibility
Reset(GlobalMersenneTwister, 42);

# Set verbose level to 0 to suppress output
VerboseLevel := 1;
info := NewInfoClass("Info");
SetInfoLevel(info, VerboseLevel);

# Inputs
n:=20;

for k in [-1..1] do
    p:=n-1-k;
    q:=n-1+k;

    # Compute representatives
    dc_reps_and_size:=GetDoubleCosetsRepsAndSizes(p, n, q);
    Print("Arguments (n-alpha,n,n-beta) = ", [n-p,n,n-q], ". Number of double cosets: ", Size(dc_reps_and_size), "\n");

    p:=n-1+k;
    q:=n-1-k;

    # Compute representatives
    dc_reps_and_size:=GetDoubleCosetsRepsAndSizes(p, n, q);
    Print("Arguments (n-alpha,n,n-beta) = ", [n-p,n,n-q], ". Number of double cosets: ", Size(dc_reps_and_size), "\n");

od;

for k in [-2..2] do
    p:=n-2-k;
    q:=n-2+k;

    # Compute representatives
    dc_reps_and_size:=GetDoubleCosetsRepsAndSizes(p, n, q);
    Print("Arguments (n-alpha,n,n-beta) = ", [n-p,n,n-q], ". Number of double cosets: ", Size(dc_reps_and_size), "\n");

    p:=n-2+k;
    q:=n-2-k;

    # Compute representatives
    dc_reps_and_size:=GetDoubleCosetsRepsAndSizes(p, n, q);
    Print("Arguments (n-alpha,n,n-beta) = ", [n-p,n,n-q], ". Number of double cosets: ", Size(dc_reps_and_size), "\n");

od;

quit;