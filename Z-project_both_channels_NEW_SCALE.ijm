run("Z Project...", "stop=100 projection=[Average Intensity]");
run("Set Scale...", "distance=1 known=0.13079 unit=µm");
run("Scale Bar...", "width=10 height=10 location=[Lower Left] horizontal bold");
run("BMP...");
run("Close All");