"""
Association Miner component for the Smart Resume Screening System.

This module provides the AssociationMiner class that discovers frequently
co-occurring skills using the Apriori algorithm to identify skill patterns
and market trends.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)


@dataclass
class AssociationRule:
    """Represents an association rule between skill sets.
    
    Attributes:
        antecedents: Set of skills in the antecedent (if-part)
        consequents: Set of skills in the consequent (then-part)
        support: Proportion of transactions containing both antecedents and consequents
        confidence: Probability of consequents given antecedents
        lift: Ratio of observed support to expected support if independent
    """
    antecedents: frozenset
    consequents: frozenset
    support: float
    confidence: float
    lift: float


class AssociationMiner:
    """Discovers frequently co-occurring skills using Apriori algorithm.
    
    This class applies the Apriori algorithm to identify skill sets that
    frequently appear together in resumes, generating association rules
    that reveal skill patterns and market trends.
    
    Attributes:
        min_support: Minimum support threshold for frequent itemsets
        min_confidence: Minimum confidence threshold for association rules
    """
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        """Initialize with support and confidence thresholds.
        
        Args:
            min_support: Minimum support threshold (default: 0.1)
            min_confidence: Minimum confidence threshold (default: 0.5)
            
        Raises:
            ValueError: If thresholds are not in valid range (0, 1]
        """
        if not 0 < min_support <= 1:
            raise ValueError("min_support must be in range (0, 1]")
        
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must be in range (0, 1]")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        logger.info(
            f"AssociationMiner initialized with min_support={min_support}, "
            f"min_confidence={min_confidence}"
        )

    def _get_effective_min_support(self, transaction_count: int) -> float:
        """Raise support for tiny datasets to avoid single-occurrence explosions."""
        if transaction_count <= 0:
            raise ValueError("transaction_count must be positive")

        # Itemsets that appear in only one resume are usually noise and can make
        # Apriori explode on small smoke-test datasets.
        return max(self.min_support, min(1.0, 2.0 / transaction_count))
    
    def mine_frequent_itemsets(self, transactions: List[List[str]]) -> pd.DataFrame:
        """Find frequent skill sets using Apriori.
        
        Transforms the transaction list into a binary matrix and applies the
        Apriori algorithm to discover itemsets that meet the minimum support
        threshold.
        
        Args:
            transactions: List of transactions, where each transaction is a list of skills
            
        Returns:
            DataFrame with columns: support, itemsets
            
        Raises:
            ValueError: If transactions list is empty or contains only empty transactions
        """
        if not transactions:
            raise ValueError("Transactions list cannot be empty")
        
        # Filter out empty transactions
        non_empty_transactions = [t for t in transactions if t]
        
        if not non_empty_transactions:
            raise ValueError("All transactions are empty")
        
        transaction_count = len(non_empty_transactions)
        effective_min_support = self._get_effective_min_support(transaction_count)

        if effective_min_support > self.min_support:
            logger.warning(
                "Raised effective min_support from %.3f to %.3f for %d transactions "
                "so Apriori ignores single-occurrence itemsets",
                self.min_support,
                effective_min_support,
                transaction_count
            )

        logger.info(
            f"Mining frequent itemsets from {transaction_count} transactions "
            f"with min_support={effective_min_support}"
        )
        
        # Transform transactions to binary matrix
        te = TransactionEncoder()
        te_array = te.fit(non_empty_transactions).transform(non_empty_transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)
        
        logger.debug(
            f"Transaction matrix shape: {df.shape} "
            f"({df.shape[0]} transactions × {df.shape[1]} unique skills)"
        )
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(
            df,
            min_support=effective_min_support,
            use_colnames=True,
            low_memory=True
        )
        
        if frequent_itemsets.empty:
            logger.warning(
                f"No frequent itemsets found with min_support={effective_min_support}. "
                "Consider lowering the threshold."
            )
        else:
            logger.info(
                f"Found {len(frequent_itemsets)} frequent itemsets "
                f"(support >= {effective_min_support})"
            )
        
        return frequent_itemsets
    
    def generate_rules(self, frequent_itemsets: pd.DataFrame) -> List[AssociationRule]:
        """Generate association rules with support, confidence, lift.
        
        Uses the frequent itemsets to generate association rules that meet
        the minimum confidence threshold. Calculates lift metric to identify
        rules where the consequent is more likely given the antecedent.
        
        Args:
            frequent_itemsets: DataFrame from mine_frequent_itemsets()
            
        Returns:
            List of AssociationRule objects
            
        Raises:
            ValueError: If frequent_itemsets is empty or invalid
        """
        if frequent_itemsets.empty:
            logger.warning("No frequent itemsets provided, returning empty rules list")
            return []
        
        logger.info(
            f"Generating association rules with min_confidence={self.min_confidence}"
        )
        
        # Generate rules using mlxtend
        try:
            rules_df = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
        except ValueError as e:
            logger.error(f"Failed to generate rules: {e}")
            return []
        
        if rules_df.empty:
            logger.warning(
                f"No rules found with min_confidence={self.min_confidence}. "
                "Consider lowering the threshold."
            )
            return []
        
        logger.info(
            f"Generated {len(rules_df)} association rules "
            f"(confidence >= {self.min_confidence})"
        )
        
        # Convert to AssociationRule objects
        rules = []
        for _, row in rules_df.iterrows():
            rule = AssociationRule(
                antecedents=row['antecedents'],
                consequents=row['consequents'],
                support=float(row['support']),
                confidence=float(row['confidence']),
                lift=float(row['lift'])
            )
            rules.append(rule)
        
        # Log top rules by lift
        if rules:
            top_rules = sorted(rules, key=lambda r: r.lift, reverse=True)[:5]
            logger.info("Top 5 rules by lift:")
            for i, rule in enumerate(top_rules, 1):
                logger.info(
                    f"  {i}. {set(rule.antecedents)} => {set(rule.consequents)} "
                    f"(lift={rule.lift:.2f}, conf={rule.confidence:.2f})"
                )
        
        return rules
    
    def mine_associations(
        self,
        resumes: List[Dict[str, any]]
    ) -> List[AssociationRule]:
        """Complete association mining pipeline.
        
        Transforms resumes to transactions (each resume = list of skills),
        mines frequent itemsets, and generates association rules.
        
        Args:
            resumes: List of resume dictionaries or StructuredResume objects
                     Each must have a 'normalized_skills' attribute/key
            
        Returns:
            List of AssociationRule objects
            
        Raises:
            ValueError: If resumes list is empty or skills cannot be extracted
        """
        if not resumes:
            raise ValueError("Resumes list cannot be empty")
        
        logger.info(f"Starting association mining on {len(resumes)} resumes")
        
        # Transform resumes to transactions
        transactions = []
        for i, resume in enumerate(resumes):
            # Handle both dict and object access
            if isinstance(resume, dict):
                skills = resume.get('normalized_skills', [])
            else:
                skills = getattr(resume, 'normalized_skills', [])
            
            if skills:
                transactions.append(skills)
            else:
                logger.debug(f"Resume {i} has no normalized skills, skipping")
        
        if not transactions:
            raise ValueError(
                "No valid transactions found. Ensure resumes have normalized_skills."
            )
        
        logger.info(
            f"Extracted {len(transactions)} transactions from {len(resumes)} resumes"
        )
        
        # Mine frequent itemsets
        frequent_itemsets = self.mine_frequent_itemsets(transactions)
        
        # Generate association rules
        rules = self.generate_rules(frequent_itemsets)
        
        logger.info(
            f"Association mining complete: {len(rules)} rules discovered"
        )
        
        return rules
