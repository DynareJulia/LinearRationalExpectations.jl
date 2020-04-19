DYNARE_ROOT = "/data/projects/dynare/git/preprocessor/src"
cd("models/example1")
run(`$DYNARE_ROOT/dynare_m example1.mod language=julia output=first json=compute`)
#cd("../example2")
#run(`$DYNARE_ROOT/dynare_m example2.mod language=julia output=first json=compute`)
cd("../..")
