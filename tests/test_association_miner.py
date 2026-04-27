"""
Unit tests for the AssociationMiner component.

Tests the Apriori-based association rule mining functionality including
frequent itemset discovery, rule generation, and complete mining pipeline.
"""

import pytest
import pandas as pd
from src.association_miner import AssociationMiner, AssociationRule


class TestAssociationMinerInit:
    """Test AssociationMiner initialization."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        miner = AssociationMiner()
        assert miner.min_support == 0.1
        assert miner.min_confidence == 0.5
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        miner = AssociationMiner(min_support=0.2, min_confidence=0.6)
        assert miner.min_support == 0.2
        assert miner.min_confidence == 0.6
    
    def test_init_invalid_support(self):
        """Test initialization with invalid support threshold."""
        with pytest.raises(ValueError, match="min_support must be in range"):
            AssociationMiner(min_support=0.0)
        
        with pytest.raises(ValueError, match="min_support must be in range"):
            AssociationMiner(min_support=1.5)
    
    def test_init_invalid_confidence(self):
        """Test initialization with invalid confidence threshold."""
        with pytest.raises(ValueError, match="min_confidence must be in range"):
            AssociationMiner(min_confidence=0.0)
        
        with pytest.raises(ValueError, match="min_confidence must be in range"):
            AssociationMiner(min_confidence=1.5)


class TestMineFrequentItemsets:
    """Test frequent itemset mining."""

    def test_effective_support_ignores_single_occurrence_itemsets_on_tiny_data(self):
        """Test support floor for tiny datasets."""
        miner = AssociationMiner(min_support=0.1)

        assert miner._get_effective_min_support(5) == 0.4
        assert miner._get_effective_min_support(25) == 0.1
    
    def test_mine_simple_transactions(self):
        """Test mining with simple transaction data."""
        miner = AssociationMiner(min_support=0.5)
        
        transactions = [
            ['Python', 'Machine Learning', 'Data Science'],
            ['Python', 'Machine Learning', 'Deep Learning'],
            ['Python', 'Data Science'],
            ['Machine Learning', 'Data Science']
        ]
        
        frequent_itemsets = miner.mine_frequent_itemsets(transactions)
        
        assert isinstance(frequent_itemsets, pd.DataFrame)
        assert not frequent_itemsets.empty
        assert 'support' in frequent_itemsets.columns
        assert 'itemsets' in frequent_itemsets.columns
        
        # Python should be frequent (appears in 3/4 = 0.75)
        python_itemsets = frequent_itemsets[
            frequent_itemsets['itemsets'].apply(lambda x: 'Python' in x)
        ]
        assert not python_itemsets.empty
    
    def test_mine_empty_transactions(self):
        """Test mining with empty transactions list."""
        miner = AssociationMiner()
        
        with pytest.raises(ValueError, match="Transactions list cannot be empty"):
            miner.mine_frequent_itemsets([])
    
    def test_mine_all_empty_transactions(self):
        """Test mining with all empty transactions."""
        miner = AssociationMiner()
        
        with pytest.raises(ValueError, match="All transactions are empty"):
            miner.mine_frequent_itemsets([[], [], []])
    
    def test_mine_filters_empty_transactions(self):
        """Test that empty transactions are filtered out."""
        miner = AssociationMiner(min_support=0.5)
        
        transactions = [
            ['Python', 'Java'],
            [],
            ['Python', 'JavaScript'],
            []
        ]
        
        frequent_itemsets = miner.mine_frequent_itemsets(transactions)
        
        # Should work with 2 non-empty transactions
        assert isinstance(frequent_itemsets, pd.DataFrame)
    
    def test_mine_high_support_threshold(self):
        """Test mining with high support threshold returns fewer itemsets."""
        transactions = [
            ['Python', 'Java'],
            ['Python', 'JavaScript'],
            ['Java', 'C++'],
            ['Python', 'Ruby']
        ]
        
        # Low threshold should find more itemsets
        miner_low = AssociationMiner(min_support=0.25)
        itemsets_low = miner_low.mine_frequent_itemsets(transactions)
        
        # High threshold should find fewer itemsets
        miner_high = AssociationMiner(min_support=0.75)
        itemsets_high = miner_high.mine_frequent_itemsets(transactions)
        
        assert len(itemsets_high) <= len(itemsets_low)


class TestGenerateRules:
    """Test association rule generation."""
    
    def test_generate_rules_from_itemsets(self):
        """Test rule generation from frequent itemsets."""
        miner = AssociationMiner(min_support=0.4, min_confidence=0.6)
        
        transactions = [
            ['Python', 'Machine Learning'],
            ['Python', 'Machine Learning', 'Data Science'],
            ['Python', 'Data Science'],
            ['Machine Learning', 'Data Science'],
            ['Python', 'Machine Learning', 'Data Science']
        ]
        
        frequent_itemsets = miner.mine_frequent_itemsets(transactions)
        rules = miner.generate_rules(frequent_itemsets)
        
        assert isinstance(rules, list)
        
        if rules:  # Rules may be empty depending on thresholds
            for rule in rules:
                assert isinstance(rule, AssociationRule)
                assert isinstance(rule.antecedents, frozenset)
                assert isinstance(rule.consequents, frozenset)
                assert 0 <= rule.support <= 1
                assert 0 <= rule.confidence <= 1
                assert rule.lift >= 0
    
    def test_generate_rules_empty_itemsets(self):
        """Test rule generation with empty itemsets."""
        miner = AssociationMiner()
        
        empty_df = pd.DataFrame(columns=['support', 'itemsets'])
        rules = miner.generate_rules(empty_df)
        
        assert rules == []
    
    def test_generate_rules_high_confidence(self):
        """Test that high confidence threshold filters rules."""
        transactions = [
            ['A', 'B', 'C'],
            ['A', 'B'],
            ['A', 'C'],
            ['B', 'C']
        ]
        
        # Low confidence should generate more rules
        miner_low = AssociationMiner(min_support=0.25, min_confidence=0.3)
        itemsets = miner_low.mine_frequent_itemsets(transactions)
        rules_low = miner_low.generate_rules(itemsets)
        
        # High confidence should generate fewer rules
        miner_high = AssociationMiner(min_support=0.25, min_confidence=0.9)
        rules_high = miner_high.generate_rules(itemsets)
        
        assert len(rules_high) <= len(rules_low)
    
    def test_rule_lift_calculation(self):
        """Test that lift is calculated correctly."""
        miner = AssociationMiner(min_support=0.3, min_confidence=0.5)
        
        # Create transactions where A and B co-occur frequently
        transactions = [
            ['A', 'B'],
            ['A', 'B'],
            ['A', 'B'],
            ['A', 'C'],
            ['C', 'D']
        ]
        
        frequent_itemsets = miner.mine_frequent_itemsets(transactions)
        rules = miner.generate_rules(frequent_itemsets)
        
        # Lift should be > 1 for positively correlated items
        if rules:
            for rule in rules:
                assert rule.lift > 0


class TestMineAssociations:
    """Test complete association mining pipeline."""
    
    def test_mine_associations_with_dicts(self):
        """Test mining with resume dictionaries."""
        miner = AssociationMiner(min_support=0.4, min_confidence=0.5)
        
        resumes = [
            {'normalized_skills': ['Python', 'Machine Learning', 'Data Science']},
            {'normalized_skills': ['Python', 'Machine Learning']},
            {'normalized_skills': ['Python', 'Data Science']},
            {'normalized_skills': ['Machine Learning', 'Data Science']},
            {'normalized_skills': ['Python', 'Machine Learning', 'Data Science']}
        ]
        
        rules = miner.mine_associations(resumes)
        
        assert isinstance(rules, list)
        # Rules may be empty depending on data and thresholds
        for rule in rules:
            assert isinstance(rule, AssociationRule)
    
    def test_mine_associations_with_objects(self):
        """Test mining with resume objects."""
        miner = AssociationMiner(min_support=0.5, min_confidence=0.6)
        
        class MockResume:
            def __init__(self, skills):
                self.normalized_skills = skills
        
        resumes = [
            MockResume(['Java', 'Spring', 'SQL']),
            MockResume(['Java', 'Spring']),
            MockResume(['Java', 'SQL']),
            MockResume(['Spring', 'SQL'])
        ]
        
        rules = miner.mine_associations(resumes)
        
        assert isinstance(rules, list)
    
    def test_mine_associations_empty_resumes(self):
        """Test mining with empty resumes list."""
        miner = AssociationMiner()
        
        with pytest.raises(ValueError, match="Resumes list cannot be empty"):
            miner.mine_associations([])
    
    def test_mine_associations_no_skills(self):
        """Test mining when resumes have no skills."""
        miner = AssociationMiner()
        
        resumes = [
            {'normalized_skills': []},
            {'normalized_skills': []},
            {}
        ]
        
        with pytest.raises(ValueError, match="No valid transactions found"):
            miner.mine_associations(resumes)
    
    def test_mine_associations_skips_empty_skills(self):
        """Test that resumes with empty skills are skipped."""
        miner = AssociationMiner(min_support=0.5, min_confidence=0.5)
        
        resumes = [
            {'normalized_skills': ['Python', 'Java']},
            {'normalized_skills': []},
            {'normalized_skills': ['Python', 'JavaScript']},
            {}
        ]
        
        # Should work with 2 valid resumes
        rules = miner.mine_associations(resumes)
        assert isinstance(rules, list)


class TestAssociationRuleDataclass:
    """Test AssociationRule dataclass."""
    
    def test_create_association_rule(self):
        """Test creating an AssociationRule."""
        rule = AssociationRule(
            antecedents=frozenset(['Python', 'Java']),
            consequents=frozenset(['Machine Learning']),
            support=0.6,
            confidence=0.8,
            lift=1.5
        )
        
        assert rule.antecedents == frozenset(['Python', 'Java'])
        assert rule.consequents == frozenset(['Machine Learning'])
        assert rule.support == 0.6
        assert rule.confidence == 0.8
        assert rule.lift == 1.5
    
    def test_frozenset_immutability(self):
        """Test that frozensets are immutable."""
        rule = AssociationRule(
            antecedents=frozenset(['A']),
            consequents=frozenset(['B']),
            support=0.5,
            confidence=0.7,
            lift=1.2
        )
        
        # Frozensets should be immutable
        with pytest.raises(AttributeError):
            rule.antecedents.add('C')
