#This bash script runs modeling_nn.py on data in all folders 
for folder in baseline_agg baseline_BC baseline_GD baseline_MB baseline_PM proposed_model
do
    echo $folder
    python modeling_nn.py -folder $folder
done

