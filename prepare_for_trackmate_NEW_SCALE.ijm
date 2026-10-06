run("Concatenate...");
run("Set Scale...", "distance=1 known=0.13079 unit=µm global");
run("Properties...", "channels=1 slices=1 frames=7500 pixel_width=0.13079 pixel_height=0.13079 voxel_depth=0.13079 frame=[25 ms] invert global");
//if (isOpen("ROI Manager")){
//	roiManager("Select", 0);
//}
run("TrackMate");