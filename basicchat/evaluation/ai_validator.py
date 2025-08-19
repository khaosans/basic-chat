"""
AI Self-Reflection and Output Validation System

This module provides comprehensive AI self-validation capabilities including:
- Output quality assessment
- Error detection and correction
- Content verification
- Response improvement suggestions
- Automatic fixing of common issues
"""

import logging
import re
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from basicchat.core.config import DEFAULT_MODEL, OLLAMA_API_URL

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Different levels of validation intensity"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"

class IssueType(Enum):
    """Types of issues that can be detected"""
    FACTUAL_ERROR = "factual_error"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    INCOMPLETE_ANSWER = "incomplete_answer"
    FORMATTING_ERROR = "formatting_error"
    GRAMMAR_ERROR = "grammar_error"
    CLARITY_ISSUE = "clarity_issue"
    RELEVANCE_ISSUE = "relevance_issue"
    BIAS_DETECTED = "bias_detected"
    HARMFUL_CONTENT = "harmful_content"

@dataclass
class ValidationIssue:
    """Represents a detected issue in the AI output"""
    issue_type: IssueType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    location: str  # where in the text the issue occurs
    suggested_fix: str
    confidence: float  # 0.0 to 1.0

@dataclass
class ValidationResult:
    """Result of AI output validation"""
    original_output: str
    quality_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue]
    improved_output: Optional[str] = None
    validation_notes: str = ""
    processing_time: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STANDARD

class AIValidator:
    """AI Self-Reflection and Output Validation System"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the AI validator"""
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_URL.replace("/api", "")
        )
        logger.info(f"AIValidator initialized with model: {model_name}")
    
    def validate_output(
        self, 
        output: str, 
        original_question: str = "", 
        context: str = "",
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResult:
        """
        Comprehensive validation of AI output
        
        Args:
            output: The AI-generated output to validate
            original_question: The original user question
            context: Any relevant context
            validation_level: Intensity of validation
            
        Returns:
            ValidationResult with quality assessment and suggestions
        """
        start_time = time.time()
        logger.info(f"Starting {validation_level.value} validation of output")
        
        try:
            # Step 1: Basic quality assessment
            quality_score = self._assess_quality(output, original_question, context)
            
            # Step 2: Detect issues
            issues = self._detect_issues(output, original_question, context, validation_level)
            
            # Step 3: Generate improved output if issues found
            improved_output = None
            if issues and any(issue.severity in ["medium", "high", "critical"] for issue in issues):
                improved_output = self._generate_improved_output(
                    output, original_question, context, issues
                )
            
            # Step 4: Generate validation notes
            validation_notes = self._generate_validation_notes(quality_score, issues)
            
            processing_time = time.time() - start_time
            
            result = ValidationResult(
                original_output=output,
                quality_score=quality_score,
                issues=issues,
                improved_output=improved_output,
                validation_notes=validation_notes,
                processing_time=processing_time,
                validation_level=validation_level
            )
            
            logger.info(f"Validation completed in {processing_time:.2f}s, quality score: {quality_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                original_output=output,
                quality_score=0.0,
                issues=[ValidationIssue(
                    issue_type=IssueType.FACTUAL_ERROR,
                    severity="high",
                    description=f"Validation system error: {e}",
                    location="system",
                    suggested_fix="Manual review required",
                    confidence=1.0
                )],
                validation_notes=f"Validation system encountered an error: {e}",
                processing_time=time.time() - start_time,
                validation_level=validation_level
            )
    
    def _assess_quality(self, output: str, question: str, context: str) -> float:
        """Assess overall quality of the output"""
        
        prompt = f"""
        As an AI quality assessor, evaluate the following response on a scale of 0.0 to 1.0.
        
        ORIGINAL QUESTION: {question}
        CONTEXT: {context}
        
        RESPONSE TO EVALUATE:
        {output}
        
        Evaluate based on:
        1. Accuracy and factual correctness
        2. Completeness of the answer
        3. Clarity and readability
        4. Relevance to the question
        5. Logical consistency
        
        Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.85)
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract numeric score
            score_match = re.search(r'(\d+\.?\d*)', content)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            else:
                logger.warning("Could not extract quality score, defaulting to 0.5")
                return 0.5
                
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5
    
    def _detect_issues(
        self, 
        output: str, 
        question: str, 
        context: str, 
        validation_level: ValidationLevel
    ) -> List[ValidationIssue]:
        """Detect various types of issues in the output"""
        
        issues = []
        
        # Basic checks (always performed)
        issues.extend(self._check_basic_issues(output))
        
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.CRITICAL]:
            issues.extend(self._check_content_issues(output, question, context))
        
        if validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.CRITICAL]:
            issues.extend(self._check_advanced_issues(output, question, context))
        
        if validation_level == ValidationLevel.CRITICAL:
            issues.extend(self._check_critical_issues(output, question, context))
        
        return issues
    
    def _check_basic_issues(self, output: str) -> List[ValidationIssue]:
        """Check for basic formatting and structural issues"""
        issues = []
        
        # Check for empty or very short responses
        if not output.strip():
            issues.append(ValidationIssue(
                issue_type=IssueType.INCOMPLETE_ANSWER,
                severity="critical",
                description="Output is empty",
                location="entire response",
                suggested_fix="Generate a proper response to the question",
                confidence=1.0
            ))
        elif len(output.strip()) < 20:
            issues.append(ValidationIssue(
                issue_type=IssueType.INCOMPLETE_ANSWER,
                severity="high",
                description="Response is too short and likely incomplete",
                location="entire response",
                suggested_fix="Provide a more detailed and complete answer",
                confidence=0.9
            ))
        
        # Check for formatting issues
        if '**' in output and output.count('**') % 2 != 0:
            issues.append(ValidationIssue(
                issue_type=IssueType.FORMATTING_ERROR,
                severity="low",
                description="Unmatched markdown bold formatting",
                location="markdown formatting",
                suggested_fix="Ensure all ** bold markers are properly paired",
                confidence=0.8
            ))
        
        # Check for repeated content
        sentences = output.split('.')
        if len(sentences) > 2:
            for i, sentence in enumerate(sentences[:-1]):
                for j, other_sentence in enumerate(sentences[i+1:], i+1):
                    if sentence.strip() and len(sentence.strip()) > 10:
                        similarity = self._calculate_text_similarity(sentence.strip(), other_sentence.strip())
                        if similarity > 0.8:
                            issues.append(ValidationIssue(
                                issue_type=IssueType.CLARITY_ISSUE,
                                severity="medium",
                                description="Detected repeated or very similar content",
                                location=f"sentences {i+1} and {j+1}",
                                suggested_fix="Remove redundant content and improve flow",
                                confidence=0.7
                            ))
                            break
        
        return issues
    
    def _check_content_issues(self, output: str, question: str, context: str) -> List[ValidationIssue]:
        """Check for content-related issues using AI analysis"""
        
        prompt = f"""
        As an AI content reviewer, analyze the following response for potential issues.
        
        ORIGINAL QUESTION: {question}
        CONTEXT: {context}
        
        RESPONSE TO ANALYZE:
        {output}
        
        Check for:
        1. Factual accuracy
        2. Logical consistency
        3. Completeness (does it fully answer the question?)
        4. Relevance to the question
        5. Clarity and coherence
        
        For each issue found, provide:
        - Type: [factual_error|logical_inconsistency|incomplete_answer|relevance_issue|clarity_issue]
        - Severity: [low|medium|high|critical]
        - Description: Brief description of the issue
        - Location: Where in the text the issue occurs
        - Fix: Suggested improvement
        
        Format as JSON array:
        [
            {{
                "type": "issue_type",
                "severity": "severity_level",
                "description": "issue description",
                "location": "where in text",
                "fix": "suggested fix"
            }}
        ]
        
        If no issues found, return: []
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            issues_data = self._extract_json_from_text(content)
            if not issues_data:
                return []
            
            issues = []
            for issue_data in issues_data:
                try:
                    issue_type_map = {
                        "factual_error": IssueType.FACTUAL_ERROR,
                        "logical_inconsistency": IssueType.LOGICAL_INCONSISTENCY,
                        "incomplete_answer": IssueType.INCOMPLETE_ANSWER,
                        "relevance_issue": IssueType.RELEVANCE_ISSUE,
                        "clarity_issue": IssueType.CLARITY_ISSUE
                    }
                    
                    issue_type = issue_type_map.get(issue_data.get("type", ""), IssueType.CLARITY_ISSUE)
                    
                    issues.append(ValidationIssue(
                        issue_type=issue_type,
                        severity=issue_data.get("severity", "medium"),
                        description=issue_data.get("description", "Unknown issue"),
                        location=issue_data.get("location", "unknown"),
                        suggested_fix=issue_data.get("fix", "Manual review needed"),
                        confidence=0.75
                    ))
                except Exception as e:
                    logger.warning(f"Could not parse issue data: {e}")
                    continue
            
            return issues
            
        except Exception as e:
            logger.error(f"Content issue detection failed: {e}")
            return []
    
    def _check_advanced_issues(self, output: str, question: str, context: str) -> List[ValidationIssue]:
        """Check for advanced issues like bias, tone, etc."""
        issues = []
        
        # Check for potential bias indicators
        bias_keywords = [
            "obviously", "clearly", "everyone knows", "it's common sense",
            "all people", "never", "always", "impossible", "definitely"
        ]
        
        for keyword in bias_keywords:
            if keyword.lower() in output.lower():
                issues.append(ValidationIssue(
                    issue_type=IssueType.BIAS_DETECTED,
                    severity="low",
                    description=f"Potential bias indicator: '{keyword}'",
                    location=f"keyword: {keyword}",
                    suggested_fix="Consider using more neutral language",
                    confidence=0.6
                ))
        
        return issues
    
    def _check_critical_issues(self, output: str, question: str, context: str) -> List[ValidationIssue]:
        """Check for critical safety and ethical issues"""
        issues = []
        
        # Check for potentially harmful content
        harmful_indicators = [
            "violence", "harm", "illegal", "dangerous", "unsafe"
        ]
        
        for indicator in harmful_indicators:
            if indicator.lower() in output.lower():
                issues.append(ValidationIssue(
                    issue_type=IssueType.HARMFUL_CONTENT,
                    severity="critical",
                    description=f"Potential harmful content detected: '{indicator}'",
                    location=f"keyword: {indicator}",
                    suggested_fix="Review content for safety and appropriateness",
                    confidence=0.8
                ))
        
        return issues
    
    def _generate_improved_output(
        self, 
        original_output: str, 
        question: str, 
        context: str, 
        issues: List[ValidationIssue]
    ) -> str:
        """Generate an improved version of the output addressing the identified issues"""
        
        issues_summary = "\n".join([
            f"- {issue.issue_type.value}: {issue.description} (Fix: {issue.suggested_fix})"
            for issue in issues if issue.severity in ["medium", "high", "critical"]
        ])
        
        prompt = f"""
        Please improve the following AI response by addressing the identified issues.
        
        ORIGINAL QUESTION: {question}
        CONTEXT: {context}
        
        ORIGINAL RESPONSE:
        {original_output}
        
        ISSUES TO ADDRESS:
        {issues_summary}
        
        Please provide an improved response that:
        1. Addresses all the identified issues
        2. Maintains the same helpful tone
        3. Keeps the response length appropriate
        4. Ensures accuracy and completeness
        
        IMPROVED RESPONSE:
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate improved output: {e}")
            return original_output
    
    def _generate_validation_notes(self, quality_score: float, issues: List[ValidationIssue]) -> str:
        """Generate human-readable validation notes"""
        
        notes = []
        
        # Quality assessment
        if quality_score >= 0.9:
            notes.append("✅ High quality response")
        elif quality_score >= 0.7:
            notes.append("👍 Good quality response")
        elif quality_score >= 0.5:
            notes.append("⚠️ Moderate quality response")
        else:
            notes.append("❌ Low quality response")
        
        # Issue summary
        if not issues:
            notes.append("✅ No significant issues detected")
        else:
            critical_issues = [i for i in issues if i.severity == "critical"]
            high_issues = [i for i in issues if i.severity == "high"]
            medium_issues = [i for i in issues if i.severity == "medium"]
            low_issues = [i for i in issues if i.severity == "low"]
            
            if critical_issues:
                notes.append(f"🚨 {len(critical_issues)} critical issue(s) detected")
            if high_issues:
                notes.append(f"⚠️ {len(high_issues)} high priority issue(s) detected")
            if medium_issues:
                notes.append(f"📝 {len(medium_issues)} medium priority issue(s) detected")
            if low_issues:
                notes.append(f"ℹ️ {len(low_issues)} minor issue(s) detected")
        
        return " | ".join(notes)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_json_from_text(self, text: str) -> Optional[List[Dict]]:
        """Extract JSON array from text response"""
        try:
            # Try to find JSON array in the text
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return []
        except Exception as e:
            logger.warning(f"Could not extract JSON from text: {e}")
            return []

class ValidationMode(Enum):
    """Different modes for applying validation"""
    DISABLED = "disabled"
    ADVISORY = "advisory"  # Show validation results but don't auto-fix
    AUTO_FIX = "auto_fix"  # Automatically use improved output if available
    INTERACTIVE = "interactive"  # Let user choose

def create_validator_instance(model_name: str = DEFAULT_MODEL) -> AIValidator:
    """Factory function to create validator instance"""
    return AIValidator(model_name)
