/*
 *	PsFileAccessorLang.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.xml;

import java.util.HashMap;

public class PsFileAccessorLang extends PsFileAccessor {

	public static final String FileName = "PalmSecureSample.lang";

	public static final String MainDialog_Title = "MainDialog.Title";
	public static final String MainDialog_EnrollBtn = "MainDialog.EnrollBtn";
	public static final String MainDialog_VerifyBtn = "MainDialog.VerifyBtn";
	public static final String MainDialog_IdentifyBtn = "MainDialog.IdentifyBtn";
	public static final String MainDialog_CancelBtn = "MainDialog.CancelBtn";
	public static final String MainDialog_DeleteBtn = "MainDialog.DeleteBtn";
	public static final String MainDialog_ExitBtn = "MainDialog.ExitBtn";
	public static final String MainDialog_IdLbl = "MainDialog.IdLbl";
	public static final String MainDialog_IdListLbl = "MainDialog.IdListLbl";
	public static final String MainDialog_IdNumLbl = "MainDialog.IdNumLbl";
	public static final String MainDialog_SensorName = "MainDialog.SensorName";
	public static final String MainDialog_SensorName0 = "MainDialog.SensorName0";
	public static final String MainDialog_SensorName1 = "MainDialog.SensorName1";
	public static final String MainDialog_SensorName2 = "MainDialog.SensorName2";
	public static final String MainDialog_SensorName3 = "MainDialog.SensorName3";
	public static final String MainDialog_SensorName4 = "MainDialog.SensorName4";
	public static final String MainDialog_SensorName5 = "MainDialog.SensorName5";
	public static final String MainDialog_SensorName6 = "MainDialog.SensorName6";
	public static final String MainDialog_SensorName7 = "MainDialog.SensorName7";
	public static final String MainDialog_SensorName1_Compati = "MainDialog.SensorName1_Compati";
	public static final String MainDialog_SensorName1_Extend = "MainDialog.SensorName1_Extend";
	public static final String MainDialog_GuideMode0 = "MainDialog.GuideMode0";
	public static final String MainDialog_GuideMode1 = "MainDialog.GuideMode1";

	public static final String Guidance_NOTIFY_CAP_GUID_BADIMAGE = "Guidance.NOTIFY_CAP_GUID_BADIMAGE";
	public static final String Guidance_NOTIFY_CAP_GUID_MOVING = "Guidance.NOTIFY_CAP_GUID_MOVING";
	public static final String Guidance_NOTIFY_CAP_GUID_LESSINFO = "Guidance.NOTIFY_CAP_GUID_LESSINFO";
	public static final String Guidance_NOTIFY_CAP_GUID_RIGHT = "Guidance.NOTIFY_CAP_GUID_RIGHT";
	public static final String Guidance_NOTIFY_CAP_GUID_LEFT = "Guidance.NOTIFY_CAP_GUID_LEFT";
	public static final String Guidance_NOTIFY_CAP_GUID_DOWN = "Guidance.NOTIFY_CAP_GUID_DOWN";
	public static final String Guidance_NOTIFY_CAP_GUID_UP = "Guidance.NOTIFY_CAP_GUID_UP";
	public static final String Guidance_NOTIFY_CAP_GUID_FAR = "Guidance.NOTIFY_CAP_GUID_FAR";
	public static final String Guidance_NOTIFY_CAP_GUID_NEAR = "Guidance.NOTIFY_CAP_GUID_NEAR";
	public static final String Guidance_NOTIFY_CAP_GUID_CAPTURING = "Guidance.NOTIFY_CAP_GUID_CAPTURING";
	public static final String Guidance_NOTIFY_CAP_GUID_PITCH_DOWN = "Guidance.NOTIFY_CAP_GUID_PITCH_DOWN";
	public static final String Guidance_NOTIFY_CAP_GUID_PITCH_UP = "Guidance.NOTIFY_CAP_GUID_PITCH_UP";
	public static final String Guidance_NOTIFY_CAP_GUID_ROLL_RIGHT = "Guidance.NOTIFY_CAP_GUID_ROLL_RIGHT";
	public static final String Guidance_NOTIFY_CAP_GUID_ROLL_LEFT = "Guidance.NOTIFY_CAP_GUID_ROLL_LEFT";
	public static final String Guidance_NOTIFY_CAP_GUID_YAW_RIGHT = "Guidance.NOTIFY_CAP_GUID_YAW_RIGHT";
	public static final String Guidance_NOTIFY_CAP_GUID_YAW_LEFT = "Guidance.NOTIFY_CAP_GUID_YAW_LEFT";
	public static final String Guidance_NOTIFY_CAP_GUID_ADJUST_LIGHT = "Guidance.NOTIFY_CAP_GUID_ADJUST_LIGHT";
	public static final String Guidance_NOTIFY_CAP_GUID_ADJUST_NG = "Guidance.NOTIFY_CAP_GUID_ADJUST_NG";
	public static final String Guidance_NOTIFY_CAP_GUID_PHASE_END = "Guidance.NOTIFY_CAP_GUID_PHASE_END";
	public static final String Guidance_NOTIFY_CAP_GUID_NO_HANDS = "Guidance.NOTIFY_CAP_GUID_NO_HANDS";
	public static final String Guidance_NOTIFY_CAP_GUID_ROUND = "Guidance.NOTIFY_CAP_GUID_ROUND";
	public static final String Guidance_NOTIFY_CAP_GUID_START = "Guidance.NOTIFY_CAP_GUID_START";
	public static final String Guidance_NOTIFY_CAP_GUID_TOO_FAR = "Guidance.NOTIFY_CAP_GUID_TOO_FAR";
	public static final String Guidance_NOTIFY_CAP_GUID_TOO_NEAR = "Guidance.NOTIFY_CAP_GUID_TOO_NEAR";

	public static final String Guidance_IllegalId = "Guidance.IllegalId";
	public static final String Guidance_RegistId = "Guidance.RegistId";
	public static final String Guidance_MaxOver = "Guidance.MaxOver";
	public static final String Guidance_AdminMaxOver = "Guidance.AdminMaxOver";
	public static final String Guidance_SelectId = "Guidance.SelectId";
	public static final String Guidance_Unregist = "Guidance.Unregist";
	public static final String Guidance_UnregistId = "Guidance.UnregistId";
	public static final String Guidance_RetryTransaction = "Guidance.RetryTransaction";
	public static final String Guidance_EnrollmentTest = "Guidance.EnrollmentTest";

	public static final String Guidance_WorkEnrollStart = "Guidance.WorkEnrollStart";
	public static final String Guidance_WorkEnroll = "Guidance.WorkEnroll";
	public static final String Guidance_WorkEnroll3 = "Guidance.WorkEnroll3";
	public static final String Guidance_WorkEnrollTest = "Guidance.WorkEnrollTest";
	public static final String Guidance_WorkIdentifyStart = "Guidance.WorkIdentifyStart";
	public static final String Guidance_WorkIdentify = "Guidance.WorkIdentify";
	public static final String Guidance_WorkVerifyStart = "Guidance.WorkVerifyStart";
	public static final String Guidance_WorkVerify = "Guidance.WorkVerify";
	public static final String Guidance_WorkDelete = "Guidance.WorkDelete";

	public static final String CompleteMessage_EnrollOk = "CompleteMessage.EnrollOk";
	public static final String CompleteMessage_EnrollNg = "CompleteMessage.EnrollNg";
	public static final String CompleteMessage_EnrollRetry = "CompleteMessage.EnrollRetry";
	public static final String CompleteMessage_Delete = "CompleteMessage.Delete";
	public static final String CompleteMessage_VerifyOk = "CompleteMessage.VerifyOk";
	public static final String CompleteMessage_VerifyNg = "CompleteMessage.VerifyNg";
	public static final String CompleteMessage_IdentifyOk = "CompleteMessage.IdentifyOk";
	public static final String CompleteMessage_IdentifyNg = "CompleteMessage.IdentifyNg";
	public static final String CompleteMessage_EnrollCancel = "CompleteMessage.EnrollCancel";
	public static final String CompleteMessage_VerifyCancel = "CompleteMessage.VerifyCancel";
	public static final String CompleteMessage_IdentifyCancel = "CompleteMessage.IdentifyCancel";

	public static final String ConfirmMessage_Delete = "ConfirmMessage.Delete";

	public static final String ErrorMessage_LibErrorTitle = "ErrorMessage.LibErrorTitle";
	public static final String ErrorMessage_LibErrorLevel = "ErrorMessage.LibErrorLevel";
	public static final String ErrorMessage_LibErrorCode = "ErrorMessage.LibErrorCode";
	public static final String ErrorMessage_LibErrorDetail = "ErrorMessage.LibErrorDetail";
	public static final String ErrorMessage_AplErrorTitle = "ErrorMessage.AplErrorTitle";
	public static final String ErrorMessage_AplErrorBmpSave = "ErrorMessage.AplErrorBmpSave";
	public static final String ErrorMessage_AplErrorBmpDirOpen = "ErrorMessage.AplErrorBmpDirOpen";
	public static final String ErrorMessage_AplErrorLogDirOpen = "ErrorMessage.AplErrorLogDirOpen";
	public static final String ErrorMessage_AplErrorLogFileOpen = "ErrorMessage.AplErrorLogFileOpen";
	public static final String ErrorMessage_AplErrorLogFileWrite = "ErrorMessage.AplErrorLogFileWrite";
	public static final String ErrorMessage_AplErrorFileDirOpen = "ErrorMessage.AplErrorFileDirOpen";
	public static final String ErrorMessage_AplErrorFileOpen = "ErrorMessage.AplErrorFileOpen";
	public static final String ErrorMessage_AplErrorFileWrite = "ErrorMessage.AplErrorFileWrite";
	public static final String ErrorMessage_AplErrorFileDelete = "ErrorMessage.AplErrorFileDelete";
	public static final String ErrorMessage_AplErrorDataFileNotFound = "ErrorMessage.AplErrorDataFileNotFound";
	public static final String ErrorMessage_AplErrorNoData = "ErrorMessage.AplErrorNoData";
	public static final String ErrorMessage_AplErrorGuideBmpLoad = "ErrorMessage.AplErrorGuideBmpLoad";
	public static final String ErrorMessage_AplErrorsilhouetteLoad = "ErrorMessage.AplErrorsilhouetteLoad";
	public static final String ErrorMessage_AplErrorSensorNothing = "ErrorMessage.AplErrorSensorNothing";
	public static final String ErrorMessage_AplErrorSystemError = "ErrorMessage.AplErrorSystemError";

	private static PsFileAccessorLang psFileAcsLang = null;
	private HashMap<String, String> xmlData = null;

	private boolean Initialize() {

		xmlData = new HashMap<String, String>();

		try {
			ReadXML(xmlData, FileName);
		} catch(Exception e) {
			xmlData = null;
			return false;
		}

		return true;
	}

	public static PsFileAccessorLang GetInstance() {

		if ( psFileAcsLang == null ) {
			psFileAcsLang = new PsFileAccessorLang();
			if ( !psFileAcsLang.Initialize() ) {
				psFileAcsLang = null;
			}
		}

		return psFileAcsLang;
	}

	public String GetValue(String key) {

		String value = null;

		try {
			value = xmlData.get(key);
		} catch(Exception e) {
			value = null;
		}

		return value;
	}

	public void SetValue(String key, String value) {

		try {
			xmlData.put(key, value);
		} catch(Exception e) {

		}

	}

}
