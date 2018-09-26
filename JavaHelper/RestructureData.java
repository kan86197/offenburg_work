import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

import javax.naming.directory.DirContext;


class RestructureData {

	private static String labelPath = "";
	private static String imagePath = "";

	/*
	 * Set the path to be use
	 * labelPath is the directory where the .txt files are, and also is the folder
	 * where the label directories will be created
	 * 
	 * imagePath is the the directory containing the images.
	 */
	public static void SetPath(String label, String image) {
		labelPath = label; //"C:\Users\USER\Desktop\Offenburg\aufgabe_1_data\train\labels";
		imagePath = image; //"C:\\Users\USER\Desktop\Offenburg\aufgabe_1_data\train\images";
	}

	public static String processPath(String path) {
		File tester = new File(path);
		if(tester.exists())
			return path.replace("\\", "\\\\");
		else 
			return "Path doesn't exist";
	}


	public static void deleteOldFiles() {
		File labelsDir = new File(labelPath);

		/*
		 * Deleting old images from previous session
		 */
		if(labelsDir.listFiles() != null)
			for(File file: labelsDir.listFiles()) {
				if(file.isDirectory()) {
					for(File oldFile: file.listFiles()) 
						oldFile.delete();
					file.delete();
				}	else file.delete();

			}
	}

	public static void main(String[]args) {
		Object[] filesPaths = null;
		BufferedReader reader = null;


		/*
		 * Rename the paths to correct directory to restructure the data
		 * 
		 *  Will not work unless the R script to move desired samples are execute first 
		 *  
		 */

		try (Stream<Path> paths = Files.walk(Paths.get(labelPath))) {
			filesPaths = paths.toArray();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		/*
		 * Creating subfolders according to the labels
		 */
		for(int a = 0; a < filesPaths.length; a++) {
			try {
				reader = new BufferedReader(new FileReader(new File(filesPaths[a].toString())));
				String currLine = reader.readLine();
				Path p = Paths.get(labelPath, currLine);
				if(currLine.equals(null)) continue;
				Files.createDirectories(p);

			} catch (Exception e) {

			}
		}


		/*
		 * Move the images into the folders it belong
		 */
		for(int a = 0; a < filesPaths.length; a++) {
			try {
				reader = new BufferedReader(new FileReader(new File(filesPaths[a].toString())));
				String currLine = reader.readLine();
				String filePath = filesPaths[a].toString();
				String fileName = filePath.substring(filePath.lastIndexOf("\\") + 1, filePath.indexOf("."));
				File file = new File(imagePath + "\\" + fileName + ".png");
				file.renameTo(new File(labelPath + "\\" + currLine + "\\" + fileName + ".png"));
			} catch (IOException e) {

			}
		}

		try {
			reader.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


}
