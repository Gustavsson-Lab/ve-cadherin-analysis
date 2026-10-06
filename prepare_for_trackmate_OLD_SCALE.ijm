run("Concatenate...");
run("Set Scale...", "distance=1 known=0.1293 unit=µm global");
run("Properties...", "channels=1 slices=1 frames=10000 pixel_width=0.1293000 pixel_height=0.1293000 voxel_depth=0.1293000 frame=[25 ms] invert global");
if (isOpen("ROI Manager")){
	roiManager("Select", 0);
}
run("TrackMate");