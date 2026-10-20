for n in 32 64 128 256 512 1024; do
  python scripts/run_bench.py -n $n -m 2000000 --save-csv scaling_results.csv
done

python scripts/plot_bench.py scaling_results.csv
