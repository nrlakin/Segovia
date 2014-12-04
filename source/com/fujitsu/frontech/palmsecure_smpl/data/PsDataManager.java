/*
 * PsDataManager.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.data;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;

import com.fujitsu.frontech.palmsecure_smpl.exception.PsAplException;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

public class PsDataManager {

	private static PsDataManager dataManager = new PsDataManager();
	private TreeMap<String, String> userMap = new TreeMap<String, String>();
	private static final String DIRECTORY_NAME_TO_STORE_DATA = "Data";

	private PsDataManager() {
	}

	public static PsDataManager GetInstance() {

		return dataManager;
	}

	public void Ps_Sample_Apl_Java_Init(long sensorType, long guideMode) throws PsAplException {

		File dataDir = new File(DIRECTORY_NAME_TO_STORE_DATA);
		if (!dataDir.exists()) {
			if (!dataDir.mkdir()) {
				PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorFileDirOpen);
				throw pae;
			}
			else
			{
				dataDir.listFiles(new PsVeinDataFilenameFilter(sensorType, guideMode));
			}
		} else {
			File[] list = dataDir.listFiles(new PsVeinDataFilenameFilter(sensorType, guideMode));

			if (list != null) {
				for (int i = 0; i < list.length; i++) {
					String id = PsVeinDataFilenameFilter.Ps_Sample_Apl_Java_GetId(list[i]);
					userMap.put(id, list[i].getAbsolutePath());
				}
			}
		}

	}

	public void Ps_Sample_Apl_Java_Insert(String userID, byte[] data) throws PsAplException {

		File dataDir = new File(DIRECTORY_NAME_TO_STORE_DATA);
		if (!dataDir.exists()) {
			dataDir.mkdir();
		}

		// Write vein data file.
		String fileName = PsVeinDataFilenameFilter.Ps_Sample_Apl_Java_GetFileName(dataDir, userID);

		File file = new File(fileName);
		FileOutputStream outStream;
		try {
			outStream = new FileOutputStream(file);
		} catch (FileNotFoundException e) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorFileOpen);
			throw pae;
		}

		FileChannel outChannel = outStream.getChannel();
		ByteBuffer byteBuff = ByteBuffer.wrap(data);
		try {
			outChannel.write(byteBuff);
		} catch (IOException e) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorFileWrite);
			throw pae;
		} finally {
			try {
				outChannel.close();
			} catch(Exception e) {
			}
		}

		userMap.put(userID, fileName);
	}

	public ArrayList<String> Ps_Sample_Apl_Java_GetAllUserId() {

		ArrayList<String> idList = new ArrayList<String>();

		Iterator<String> itr = userMap.keySet().iterator();
        while (itr.hasNext()) {
         	String mapKey = (String) itr.next();
			idList.add(mapKey);
        }

		return idList;
	}

	public boolean Ps_Sample_Apl_Java_IsRegist(String userId) {

		String data = userMap.get(userId);
		if (data == null) {
			return false;
		}
		else {
			return true;
		}
	}

	public int Ps_Sample_Apl_Java_GetRegistNum() {

		return userMap.size();
	}

	public void Ps_Sample_Apl_Java_Delete(String userId) throws PsAplException {

		String filename = userMap.get(userId);
		if (filename == null) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorFileDelete);
			throw pae;
		}

		File file = new File(filename);
		file.delete();
		userMap.remove(userId);
	}

	public byte[] Ps_Sample_Apl_Java_GetTemplate(String userId) throws PsAplException {

		String filename = userMap.get(userId);
		if (filename == null) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorDataFileNotFound);
			throw pae;
		}

		FileInputStream inStream;
		try {
			inStream = new FileInputStream(filename);
		} catch (FileNotFoundException e) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorDataFileNotFound);
			throw pae;
		}

		FileChannel inChannel = inStream.getChannel();
		ByteBuffer byteBuff = null;
		try {
			Long fileSize = new Long(inChannel.size());
			byteBuff = ByteBuffer.allocate(fileSize.intValue());
			inChannel.read(byteBuff);
		} catch (IOException e) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorFileOpen);
			throw pae;
		} finally {
			try {
				inChannel.close();
			} catch(Exception e) {
			}
		}

		return byteBuff.array().clone();
	}
}
