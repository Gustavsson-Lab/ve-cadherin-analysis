// Select parent directory containing folders with green channel files
parentDir = getDirectory("Choose parent directory containing folders");

// Get list of subdirectories
folderList = getFileList(parentDir);

setBatchMode(true);

run("Close All");

// Loop through all subdirectories
for (i = 0; i < folderList.length; i++) {

	// Get list of files in subdirectory
	folderPath = parentDir + folderList[i];
	fileList = getFileList(folderPath);
	print("\n--- Processing folder: " + folderList[i] + " ---");
	
	// Get filename for the tiff stack
	for (j = 0; j < fileList.length; j++) {
        fileName = toLowerCase(fileList[j]);
        if (endsWith(fileName, ".nd2") || endsWith(fileName, ".ndtiff") || endsWith(fileName, ".tif") || endsWith(fileName, ".tiff")) {
            tiffFile = fileList[j];
        }
    }
	
	// Get path to image sequence
	tiffPath = folderPath + tiffFile;
	
	// Open tiff stack
    run("Bio-Formats Importer",
		"open=[" + tiffPath + "]" + 
		"autoscale " +
		"color_mode=Default " +
		"view=Hyperstack " +
		"stack_order=XYTZC");
		
	// Average first 100 frames
	run("Z Project...", "stop=100 projection=[Average Intensity]");
	
	// Close stack
	imList = getList("image.titles");
	for (k = 0; k < imList.length; k++) {
		selectImage(imList[k]);
		if (!imList[k].contains("AVG")) {
			close();
		}
	}
	
	print(imList[0]);
	
	// Set scale
	run("Set Scale...", "distance=1 known=0.13079 unit=µm");
	
	// Add scale bar
	run("Scale Bar...", "width=10 height=10 location=[Lower Left] horizontal bold");
	
	// Get image name and save as bmp
	imageName = getInfo("image.title");
//	print(imageName + " <- before '.tif' removal");
	
	// Remove ".tif" from imageName
	imageName = imageName.substring(0, imageName.length() - 4);
//	print(imageName + " <- after '.tif' removal");
	
	// If part of filepath is still in name for some reason, remove it
	if (imageName.indexOf("/") != -1) {
		imageNameNoFP = imageName.substring(0, imageName.indexOf("/"));
//		print(imageNameNoFP + " <- path removed");
	}
	else {
		imageNameNoFP = imageName
	}
	
	imageNameShort = imageNameNoFP.substring(0, imageNameNoFP.lastIndexOf("_"));
//	print(imageNameShort + " <- imageNameShort");
	
	// Save as .bmp and close image
//	print(folderPath + imageNameShort + ".bmp");
	saveAs(folderPath + imageNameShort + ".bmp"); 
	run("Close All");
	
}

print("=== PROCESSING FINISHED ===");


