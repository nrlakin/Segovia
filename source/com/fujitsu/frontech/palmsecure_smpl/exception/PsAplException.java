/*
 *	PsException.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.exception;

import com.fujitsu.frontech.palmsecure.JAVA_PvAPI_ErrorInfo;

public class PsAplException extends Exception {

	private static final long serialVersionUID = 1L;
	private String errorMsgKey = null;
	private JAVA_PvAPI_ErrorInfo errorInfo = null;

	public PsAplException() {
		setErrorMsgKey(null);
	}

	public PsAplException(String errorMsgKey) {
		setErrorMsgKey(errorMsgKey);
	}

	public PsAplException(JAVA_PvAPI_ErrorInfo errorInfo) {
		setErrorInfo(errorInfo);
	}

	public void setErrorMsgKey(String errorMsgKey) {

		this.errorMsgKey = errorMsgKey;
	}

	public String getErrorMsgKey() {

		return errorMsgKey;
	}

	public JAVA_PvAPI_ErrorInfo getErrorInfo() {

		return errorInfo;
	}

	public void setErrorInfo(JAVA_PvAPI_ErrorInfo errorInfo) {

		this.errorInfo = errorInfo;
	}
}
