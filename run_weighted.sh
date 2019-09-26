#This bash script runs modeling_nn.py on data in all folders 
for folder in proposed_model_weighted
do
    echo $folder
    python modeling_logit.py -folder $folder
    python modeling_ab.py -folder $folder
    python modeling_rf.py -folder $folder
    python modeling_nn.py -folder $folder
done

