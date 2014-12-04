/*
 * PsInputIdText.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import javax.swing.JTextField;
import javax.swing.text.AttributeSet;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;
import javax.swing.text.PlainDocument;

public class PsInputIdText extends JTextField {

	private static final long serialVersionUID = 1L;

	public PsInputIdText() {
         super();
     }

     protected Document createDefaultModel() {
         return new UpperCaseDocument();
     }

     static class UpperCaseDocument extends PlainDocument {

		private static final long serialVersionUID = 1L;

		public void insertString(int offs, String str, AttributeSet a) throws BadLocationException {


			if (str == null) {
				return;
			}

			char[] upper = str.toCharArray();
			String newStr = new String();

			for (int i = 0; i < upper.length; i++) {
				if(super.getLength() + newStr.length() >= 16) {
					break;
				}

				upper[i] = Character.toUpperCase(upper[i]);

				if((upper[i] >= '0' && upper[i] <= '9')
						|| (upper[i] >= 'A' && upper[i] <= 'Z')
						|| (upper[i] == '-') || (upper[i] == '_')) {
					newStr += upper[i];
				}
			}
			super.insertString(offs, newStr, a);
		}
	}
}
