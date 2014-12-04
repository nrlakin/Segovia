/*
 *	PsFileAccessorIni.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.xml;

import java.util.HashMap;

public class PsFileAccessorIni extends PsFileAccessor {

	public static final String FileName = "PalmSecureSample.ini";

	public static final String ApplicationKey = "ApplicationKey";
	public static final String GuideMode = "GuideMode";
	public static final String MaxResults = "MaxResults";
	public static final String NumberOfRetry = "NumberOfRetry";
	public static final String LogMode = "LogMode";
	public static final String LogFolderPath = "LogFolderPath";
	public static final String SilhouetteMode = "SilhouetteMode";
	public static final String SleepTime = "SleepTime";

    public static final String DEFAULT_ApplicationKey = "";
	public static final int DEFAULT_GuideMode = 0;
	public static final int DEFAULT_MaxResults = 2;
	public static final int DEFAULT_NumberOfRetry = 2;
    public static final int DEFAULT_LogMode = 1;
    public static final String DEFAULT_LogFolderPath = "Log";
    public static final int DEFAULT_SilhouetteMode = 1;
    public static final int DEFAULT_SleepTime = 2000;

	private static PsFileAccessorIni psFileAcsIni = null;
	private HashMap<String, String> xmlData = null;
	private HashMap<String, Integer> intData = null;

	private boolean Initialize() {

		xmlData = new HashMap<String, String>();
		intData = new HashMap<String, Integer>();

		try {
			ReadXML(xmlData, FileName);
		} catch(Exception e) {
			xmlData = null;
			intData = null;
			return false;
		}

		return true;
	}

	public static PsFileAccessorIni GetInstance() {

		if ( psFileAcsIni == null ) {
			psFileAcsIni = new PsFileAccessorIni();
			if ( !psFileAcsIni.Initialize() ) {
				psFileAcsIni = null;
			}
			if ( !psFileAcsIni.CheckAndSetValues() ) {
				psFileAcsIni = null;
			}
		}

		return psFileAcsIni;
	}

	public String GetValueString(String key) {

		String value = null;

		try {
			value = xmlData.get(key);
		} catch(Exception e) {
			value = null;
		}

		return value;
	}

	public Integer GetValueInteger(String key) {

		Integer value = null;

		try {
			value = intData.get(key);
		} catch(Exception e) {
			value = null;
		}

		return value;

	}

	private boolean CheckAndSetValues() {

		int workInt = 0;
		String workStr = null;

		if ( xmlData == null || intData == null ) {
			return false;
		}

		try {
			workStr = xmlData.get(ApplicationKey);
			if ( workStr == null || ( workStr.compareTo("") == 0 ) ) {
				xmlData.put(ApplicationKey, DEFAULT_ApplicationKey);
			}

			workInt = Integer.parseInt(xmlData.get(GuideMode));
			if ( 0 <= workInt && workInt <= 1 ) {
				intData.put(GuideMode, workInt);
			} else {
				intData.put(GuideMode, DEFAULT_GuideMode);
			}

			workInt = Integer.parseInt(xmlData.get(MaxResults));
			if ( 0 < workInt && workInt < 31 ) {
				intData.put(MaxResults, workInt);
			} else {
				intData.put(MaxResults, DEFAULT_MaxResults);
			}

			workInt = Integer.parseInt(xmlData.get(NumberOfRetry));
			if ( 0 <= workInt ) {
				intData.put(NumberOfRetry, workInt);
			} else {
				intData.put(NumberOfRetry, DEFAULT_NumberOfRetry);
			}

			workInt = Integer.parseInt(xmlData.get(LogMode));
			if ( 0 <= workInt && workInt <= 1 ) {
				intData.put(LogMode, workInt);
			} else {
				intData.put(LogMode, DEFAULT_LogMode);
			}

			workStr = xmlData.get(LogFolderPath);
			if ( workStr == null || ( workStr.compareTo("") == 0 ) ) {
				xmlData.put(LogFolderPath, DEFAULT_LogFolderPath);
			}

			workInt = Integer.parseInt(xmlData.get(SilhouetteMode));
			if ( 0 <= workInt && workInt <= 1 ) {
				intData.put(SilhouetteMode, workInt);
			} else {
				intData.put(SilhouetteMode, DEFAULT_SilhouetteMode);
			}

			workInt = Integer.parseInt(xmlData.get(SleepTime));
			if ( 0 <= workInt ) {
				intData.put(SleepTime, workInt);
			} else {
				intData.put(SleepTime, DEFAULT_SleepTime);
			}

		} catch(Exception e) {
			return false;
		}

		return true;
	}

}
