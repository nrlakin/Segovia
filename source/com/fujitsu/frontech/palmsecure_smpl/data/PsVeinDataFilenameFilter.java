/*
 *	PsVeinDataFilenameFilter.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.data;

import java.io.File;
import java.io.FilenameFilter;

public class PsVeinDataFilenameFilter implements FilenameFilter {

	private static final String VEIN_DATA_FILE_EXT = ".dat";
	private static final int ID_LENGTH = 16;
	private static String sensorType;
	private static String guideMode;

	public PsVeinDataFilenameFilter(long sensorType, long guideMode) {
		PsVeinDataFilenameFilter.sensorType = Long.toString(sensorType);
		PsVeinDataFilenameFilter.guideMode = Long.toString(guideMode);
	}

	public boolean accept(File dir, String name) {

		boolean result = false;

		String checkFormat = sensorType + guideMode + "_";

		File file = new File(dir, name);
		if (file.isFile()) {

			String fileName = file.getName();
			if (fileName.endsWith(VEIN_DATA_FILE_EXT)) {
				if (fileName.startsWith(checkFormat)) {
					if (fileName.length() <=
							(checkFormat.length() + ID_LENGTH + VEIN_DATA_FILE_EXT.length())) {
						result = true;
					}
				}
			}
		}

		return result;
	}

	public static String Ps_Sample_Apl_Java_GetId(File file) {

		String id = file.getName();								// XY_ZZZZZZZZZZZZZZZZ.dat
		id = id.substring(0, id.indexOf(VEIN_DATA_FILE_EXT));	// XY_ZZZZZZZZZZZZZZZZ
		id = id.substring(3);									// ZZZZZZZZZZZZZZZZ

		return id;
	}

	public static String Ps_Sample_Apl_Java_GetFileName(File dir, String id) {

		String header = sensorType + guideMode + "_";

		StringBuffer buff = new StringBuffer(dir.getAbsolutePath());
		buff = buff.append(File.separator);
		buff = buff.append(header);
		buff = buff.append(id);
		buff = buff.append(VEIN_DATA_FILE_EXT);

		return buff.toString();
	}
}
