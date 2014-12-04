/*
 * PsMessageDialog.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import java.awt.Component;
import javax.swing.JOptionPane;

import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;


public class PsMessageDialog extends JOptionPane {

	private static final long serialVersionUID = 1L;
	private static final String  NEW_LINE_CODE = System.getProperty("line.separator");

	private static String ErrorFormat = null;

	public static void Ps_Sample_Apl_Java_ShowErrorDialog(Component parent, String errMsgKey) {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		String message = langAcs.GetValue(PsFileAccessorLang.ErrorMessage_AplErrorTitle);
		message += NEW_LINE_CODE;
		message += langAcs.GetValue(errMsgKey);

		showMessageDialog(parent, message, "", ERROR_MESSAGE);
	}

	public static void Ps_Sample_Apl_Java_ShowErrorDialog(Component parent, JAVA_PvAPI_ErrorInfo errorInfo) {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		if (ErrorFormat == null) {
			// Error title
			ErrorFormat = langAcs.GetValue(PsFileAccessorLang.ErrorMessage_LibErrorTitle) + NEW_LINE_CODE;
			// Error level
			ErrorFormat += langAcs.GetValue(PsFileAccessorLang.ErrorMessage_LibErrorLevel)
						+ " : 0x%02x" + NEW_LINE_CODE;
			// Error code
			ErrorFormat += langAcs.GetValue(PsFileAccessorLang.ErrorMessage_LibErrorCode)
						+ " : 0x%02x" + NEW_LINE_CODE;
			// Error detail
			ErrorFormat += langAcs.GetValue(PsFileAccessorLang.ErrorMessage_LibErrorDetail)
						+ " : 0x%08x" + NEW_LINE_CODE;
			ErrorFormat += "ErrorInfo1    : 0x%08x" + NEW_LINE_CODE;
			ErrorFormat += "ErrorInfo2    : 0x%08x" + NEW_LINE_CODE;
			ErrorFormat += "ErrorInfo3[0] : 0x%08x" + NEW_LINE_CODE;
			ErrorFormat += "ErrorInfo3[1] : 0x%08x" + NEW_LINE_CODE;
			ErrorFormat += "ErrorInfo3[2] : 0x%08x" + NEW_LINE_CODE;
			ErrorFormat += "ErrorInfo3[3] : 0x%08x" + NEW_LINE_CODE;
		}

		String message = String.format(ErrorFormat,
				errorInfo.ErrorLevel,
				errorInfo.ErrorCode,
				errorInfo.ErrorDetail,
				errorInfo.ErrorInfo1,
				errorInfo.ErrorInfo2,
				errorInfo.ErrorInfo3[0],
				errorInfo.ErrorInfo3[1],
				errorInfo.ErrorInfo3[2],
				errorInfo.ErrorInfo3[3]);

		showMessageDialog(parent, message, "", ERROR_MESSAGE);
	}

	public static void Ps_Sample_Apl_Java_ShowErrorDialog(Component parent, PalmSecureException e) {

		String message = "PalmSecureException";
		message += NEW_LINE_CODE;
		message += "Error No: " + e.ErrNumber;

		showMessageDialog(parent, message, "", ERROR_MESSAGE);
	}

	public static int Ps_Sample_Apl_Java_ShowConfirmDialog(Component parent, String msgKey) {

		PsFileAccessorLang langFile = PsFileAccessorLang.GetInstance();

		String message = langFile.GetValue(msgKey);

		int action = showConfirmDialog(parent, message, "", OK_CANCEL_OPTION, WARNING_MESSAGE);

		return action;
	}
}
