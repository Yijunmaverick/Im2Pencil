cd pretrained_models

mkdir edge_model
cd edge_model
wget -c http://vllab1.ucmerced.edu/~yli62/Im2Pencil/edge_model/latest_net_G.pth
cd ..

mkdir shading_model
cd shading_model
wget -c http://vllab1.ucmerced.edu/~yli62/Im2Pencil/shading_model/latest_net_G.pth

cd ../..
