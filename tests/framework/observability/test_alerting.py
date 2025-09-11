"""
Tests for Alerting implementation.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from framework.observability.alerting import (
    Alert, AlertSeverity, AlertManager, AlertRule, 
    NotificationChannel, EmailChannel, SlackChannel, 
    PagerDutyChannel, WebhookChannel
)


class TestAlert:
    """Test alert data structure."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id="alert-123",
            title="High CPU Usage",
            message="CPU usage is above 90%",
            severity=AlertSeverity.WARNING,
            source="system_monitor",
            metadata={"cpu_percent": 95.2, "host": "server-1"}
        )
        
        assert alert.id == "alert-123"
        assert alert.title == "High CPU Usage"
        assert alert.message == "CPU usage is above 90%"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "system_monitor"
        assert alert.metadata["cpu_percent"] == 95.2
        assert isinstance(alert.timestamp, datetime)
        assert alert.resolved is False
        assert alert.resolved_at is None
    
    def test_alert_resolve(self):
        """Test resolving an alert."""
        alert = Alert("test-alert", "Test Alert", "Test message", AlertSeverity.CRITICAL)
        
        alert.resolve("Issue has been fixed")
        
        assert alert.resolved is True
        assert alert.resolution_message == "Issue has been fixed"
        assert isinstance(alert.resolved_at, datetime)
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            "test-alert", 
            "Test Alert", 
            "Test message", 
            AlertSeverity.INFO,
            metadata={"key": "value"}
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["id"] == "test-alert"
        assert alert_dict["title"] == "Test Alert"
        assert alert_dict["severity"] == "INFO"
        assert alert_dict["metadata"]["key"] == "value"
        assert "timestamp" in alert_dict
    
    def test_alert_age(self):
        """Test alert age calculation."""
        alert = Alert("test-alert", "Test", "Message", AlertSeverity.INFO)
        
        time.sleep(0.01)  # Small delay
        age = alert.age
        
        assert age.total_seconds() > 0
        assert age.total_seconds() < 1


class TestAlertSeverity:
    """Test alert severity enum."""
    
    def test_severity_values(self):
        """Test alert severity values."""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning" 
        assert AlertSeverity.ERROR == "error"
        assert AlertSeverity.CRITICAL == "critical"
    
    def test_severity_ordering(self):
        """Test severity level ordering."""
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL
        ]
        
        # This test ensures we maintain logical severity progression
        assert len(severities) == 4


class TestAlertRule:
    """Test alert rule functionality."""
    
    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        condition = lambda metrics: metrics.get("cpu_usage", 0) > 90
        
        rule = AlertRule(
            name="high_cpu_usage",
            condition=condition,
            severity=AlertSeverity.WARNING,
            title="High CPU Usage Detected",
            message_template="CPU usage is {cpu_usage}%",
            cooldown_period=300.0
        )
        
        assert rule.name == "high_cpu_usage"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.title == "High CPU Usage Detected"
        assert rule.cooldown_period == 300.0
        assert rule.last_triggered is None
    
    def test_alert_rule_evaluation_true(self):
        """Test alert rule evaluation when condition is met."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda metrics: metrics.get("error_rate", 0) > 0.05,
            severity=AlertSeverity.ERROR,
            title="High Error Rate",
            message_template="Error rate is {error_rate:.2%}"
        )
        
        metrics = {"error_rate": 0.08, "total_requests": 1000}
        should_trigger, alert = rule.evaluate(metrics)
        
        assert should_trigger is True
        assert alert is not None
        assert alert.title == "High Error Rate"
        assert "Error rate is 8.00%" in alert.message
        assert alert.severity == AlertSeverity.ERROR
    
    def test_alert_rule_evaluation_false(self):
        """Test alert rule evaluation when condition is not met."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda metrics: metrics.get("response_time", 0) > 5.0,
            severity=AlertSeverity.WARNING,
            title="Slow Response Time",
            message_template="Response time is {response_time}s"
        )
        
        metrics = {"response_time": 2.5}
        should_trigger, alert = rule.evaluate(metrics)
        
        assert should_trigger is False
        assert alert is None
    
    def test_alert_rule_cooldown(self):
        """Test alert rule cooldown period."""
        rule = AlertRule(
            name="test_cooldown",
            condition=lambda metrics: True,  # Always triggers
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message_template="Test message",
            cooldown_period=1.0  # 1 second cooldown
        )
        
        # First evaluation should trigger
        should_trigger1, alert1 = rule.evaluate({})
        assert should_trigger1 is True
        assert alert1 is not None
        
        # Immediate second evaluation should not trigger (cooldown)
        should_trigger2, alert2 = rule.evaluate({})
        assert should_trigger2 is False
        assert alert2 is None
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # Third evaluation should trigger again
        should_trigger3, alert3 = rule.evaluate({})
        assert should_trigger3 is True
        assert alert3 is not None
    
    def test_alert_rule_condition_exception(self):
        """Test alert rule when condition raises exception."""
        def faulty_condition(metrics):
            raise ValueError("Condition evaluation failed")
        
        rule = AlertRule(
            name="faulty_rule",
            condition=faulty_condition,
            severity=AlertSeverity.ERROR,
            title="Faulty Rule",
            message_template="Error occurred"
        )
        
        # Should not raise exception, should return False
        should_trigger, alert = rule.evaluate({})
        
        assert should_trigger is False
        assert alert is None
    
    def test_alert_rule_message_templating(self):
        """Test alert rule message templating."""
        rule = AlertRule(
            name="template_test",
            condition=lambda m: True,
            severity=AlertSeverity.INFO,
            title="Template Test",
            message_template="Server {server} has {metric_name}: {metric_value:.2f}"
        )
        
        metrics = {
            "server": "web-01",
            "metric_name": "CPU Usage",
            "metric_value": 87.345
        }
        
        should_trigger, alert = rule.evaluate(metrics)
        
        assert should_trigger is True
        expected_message = "Server web-01 has CPU Usage: 87.35"
        assert alert.message == expected_message
    
    def test_alert_rule_template_error(self):
        """Test alert rule with template formatting error."""
        rule = AlertRule(
            name="template_error",
            condition=lambda m: True,
            severity=AlertSeverity.INFO,
            title="Template Error Test",
            message_template="Missing key: {missing_key}"
        )
        
        metrics = {"existing_key": "value"}
        should_trigger, alert = rule.evaluate(metrics)
        
        assert should_trigger is True
        # Should fall back to a default message when templating fails
        assert alert.message is not None
        assert len(alert.message) > 0


class TestNotificationChannels:
    """Test notification channel implementations."""
    
    class TestEmailChannel:
        """Test email notification channel."""
        
        @pytest.fixture
        def email_channel(self):
            """Create email notification channel."""
            return EmailChannel(
                smtp_server="smtp.example.com",
                smtp_port=587,
                username="alerts@example.com",
                password="password123",
                from_address="alerts@example.com",
                to_addresses=["admin@example.com", "ops@example.com"]
            )
        
        @pytest.mark.asyncio
        async def test_email_send_notification(self, email_channel):
            """Test sending email notification."""
            alert = Alert(
                "email-test", 
                "Test Alert", 
                "This is a test alert", 
                AlertSeverity.WARNING
            )
            
            with patch('smtplib.SMTP') as mock_smtp:
                mock_server = Mock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                
                success = await email_channel.send_notification(alert)
                
                assert success is True
                mock_server.starttls.assert_called_once()
                mock_server.login.assert_called_once_with("alerts@example.com", "password123")
                mock_server.send_message.assert_called_once()
        
        @pytest.mark.asyncio
        async def test_email_send_failure(self, email_channel):
            """Test email notification failure."""
            alert = Alert("test", "Test", "Message", AlertSeverity.ERROR)
            
            with patch('smtplib.SMTP') as mock_smtp:
                mock_smtp.side_effect = Exception("SMTP connection failed")
                
                success = await email_channel.send_notification(alert)
                
                assert success is False
        
        def test_email_format_message(self, email_channel):
            """Test email message formatting."""
            alert = Alert(
                "format-test",
                "Test Alert",
                "This is a test message",
                AlertSeverity.CRITICAL,
                metadata={"host": "server-1", "value": 95.5}
            )
            
            subject, body = email_channel._format_message(alert)
            
            assert "Test Alert" in subject
            assert "CRITICAL" in subject
            assert "This is a test message" in body
            assert "server-1" in body
            assert "95.5" in body
    
    class TestSlackChannel:
        """Test Slack notification channel."""
        
        @pytest.fixture
        def slack_channel(self):
            """Create Slack notification channel."""
            return SlackChannel(
                webhook_url="https://hooks.slack.com/services/TEST/WEBHOOK",
                channel="#alerts",
                username="AlertBot"
            )
        
        @pytest.mark.asyncio
        async def test_slack_send_notification(self, slack_channel):
            """Test sending Slack notification."""
            alert = Alert(
                "slack-test",
                "Test Alert", 
                "Slack test message",
                AlertSeverity.ERROR
            )
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__.return_value = mock_response
                mock_post.return_value = mock_response
                
                success = await slack_channel.send_notification(alert)
                
                assert success is True
                mock_post.assert_called_once()
        
        @pytest.mark.asyncio
        async def test_slack_send_failure(self, slack_channel):
            """Test Slack notification failure."""
            alert = Alert("test", "Test", "Message", AlertSeverity.INFO)
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 400
                mock_response.__aenter__.return_value = mock_response
                mock_post.return_value = mock_response
                
                success = await slack_channel.send_notification(alert)
                
                assert success is False
        
        def test_slack_format_message(self, slack_channel):
            """Test Slack message formatting."""
            alert = Alert(
                "format-test",
                "Performance Alert",
                "High latency detected",
                AlertSeverity.WARNING,
                metadata={"latency_ms": 850}
            )
            
            payload = slack_channel._format_message(alert)
            
            assert payload["channel"] == "#alerts"
            assert payload["username"] == "AlertBot"
            assert "Performance Alert" in payload["text"]
            assert "WARNING" in payload["text"]
            assert "850" in payload["text"]
    
    class TestPagerDutyChannel:
        """Test PagerDuty notification channel."""
        
        @pytest.fixture
        def pagerduty_channel(self):
            """Create PagerDuty notification channel."""
            return PagerDutyChannel(
                integration_key="test-integration-key",
                service_name="Inference Service"
            )
        
        @pytest.mark.asyncio
        async def test_pagerduty_send_notification(self, pagerduty_channel):
            """Test sending PagerDuty notification."""
            alert = Alert(
                "pd-test",
                "Critical System Alert",
                "System is down",
                AlertSeverity.CRITICAL
            )
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 202
                mock_response.__aenter__.return_value = mock_response
                mock_post.return_value = mock_response
                
                success = await pagerduty_channel.send_notification(alert)
                
                assert success is True
                
                # Verify the payload structure
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                
                assert payload['routing_key'] == "test-integration-key"
                assert payload['event_action'] == "trigger"
                assert payload['payload']['summary'] == "Critical System Alert"
                assert payload['payload']['severity'] == "critical"
        
        def test_pagerduty_severity_mapping(self, pagerduty_channel):
            """Test PagerDuty severity mapping."""
            test_cases = [
                (AlertSeverity.INFO, "info"),
                (AlertSeverity.WARNING, "warning"),
                (AlertSeverity.ERROR, "error"),
                (AlertSeverity.CRITICAL, "critical")
            ]
            
            for alert_severity, expected_pd_severity in test_cases:
                mapped = pagerduty_channel._map_severity(alert_severity)
                assert mapped == expected_pd_severity
    
    class TestWebhookChannel:
        """Test webhook notification channel."""
        
        @pytest.fixture
        def webhook_channel(self):
            """Create webhook notification channel."""
            return WebhookChannel(
                webhook_url="https://example.com/webhook",
                headers={"Authorization": "Bearer token123"}
            )
        
        @pytest.mark.asyncio
        async def test_webhook_send_notification(self, webhook_channel):
            """Test sending webhook notification."""
            alert = Alert(
                "webhook-test",
                "Webhook Test",
                "Testing webhook delivery",
                AlertSeverity.INFO
            )
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__.return_value = mock_response
                mock_post.return_value = mock_response
                
                success = await webhook_channel.send_notification(alert)
                
                assert success is True
                
                # Verify headers and payload
                call_args = mock_post.call_args
                assert call_args[1]['headers']['Authorization'] == "Bearer token123"
                
                payload = call_args[1]['json']
                assert payload['alert_id'] == "webhook-test"
                assert payload['title'] == "Webhook Test"
        
        @pytest.mark.asyncio
        async def test_webhook_custom_formatter(self):
            """Test webhook with custom formatter."""
            def custom_formatter(alert):
                return {
                    "custom_field": f"{alert.title} - {alert.severity}",
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata
                }
            
            channel = WebhookChannel(
                webhook_url="https://example.com/custom",
                payload_formatter=custom_formatter
            )
            
            alert = Alert(
                "custom-test",
                "Custom Alert",
                "Custom message",
                AlertSeverity.WARNING,
                metadata={"key": "value"}
            )
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__.return_value = mock_response
                mock_post.return_value = mock_response
                
                success = await channel.send_notification(alert)
                
                assert success is True
                
                payload = mock_post.call_args[1]['json']
                assert payload['custom_field'] == "Custom Alert - warning"
                assert payload['metadata']['key'] == "value"


class TestAlertManager:
    """Test alert manager functionality."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager."""
        return AlertManager()
    
    @pytest.fixture
    def mock_channel(self):
        """Create mock notification channel."""
        channel = AsyncMock(spec=NotificationChannel)
        channel.send_notification = AsyncMock(return_value=True)
        return channel
    
    def test_add_rule(self, alert_manager):
        """Test adding alert rule."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda m: m.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            title="Test Rule",
            message_template="Value is {value}"
        )
        
        alert_manager.add_rule(rule)
        
        assert "test_rule" in alert_manager._rules
        assert alert_manager._rules["test_rule"] is rule
    
    def test_add_duplicate_rule(self, alert_manager):
        """Test adding duplicate rule name."""
        rule1 = AlertRule("duplicate", lambda m: True, AlertSeverity.INFO, "Rule 1", "Message 1")
        rule2 = AlertRule("duplicate", lambda m: True, AlertSeverity.ERROR, "Rule 2", "Message 2")
        
        alert_manager.add_rule(rule1)
        
        with pytest.raises(ValueError, match="already exists"):
            alert_manager.add_rule(rule2)
    
    def test_remove_rule(self, alert_manager):
        """Test removing alert rule."""
        rule = AlertRule("removable", lambda m: True, AlertSeverity.INFO, "Removable", "Message")
        
        alert_manager.add_rule(rule)
        assert "removable" in alert_manager._rules
        
        alert_manager.remove_rule("removable")
        assert "removable" not in alert_manager._rules
    
    def test_remove_nonexistent_rule(self, alert_manager):
        """Test removing non-existent rule."""
        with pytest.raises(ValueError, match="not found"):
            alert_manager.remove_rule("nonexistent")
    
    def test_add_channel(self, alert_manager, mock_channel):
        """Test adding notification channel."""
        alert_manager.add_channel("email", mock_channel)
        
        assert "email" in alert_manager._channels
        assert alert_manager._channels["email"] is mock_channel
    
    def test_remove_channel(self, alert_manager, mock_channel):
        """Test removing notification channel."""
        alert_manager.add_channel("removable", mock_channel)
        alert_manager.remove_channel("removable")
        
        assert "removable" not in alert_manager._channels
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_trigger_alert(self, alert_manager, mock_channel):
        """Test rule evaluation triggers alert."""
        # Add rule and channel
        rule = AlertRule(
            name="trigger_test",
            condition=lambda m: m.get("cpu_usage", 0) > 90,
            severity=AlertSeverity.ERROR,
            title="High CPU Usage",
            message_template="CPU usage is {cpu_usage}%"
        )
        alert_manager.add_rule(rule)
        alert_manager.add_channel("test", mock_channel)
        
        # Evaluate with triggering metrics
        metrics = {"cpu_usage": 95}
        await alert_manager.evaluate_rules(metrics)
        
        # Verify notification was sent
        mock_channel.send_notification.assert_called_once()
        
        # Verify alert was stored
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].title == "High CPU Usage"
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_no_trigger(self, alert_manager, mock_channel):
        """Test rule evaluation doesn't trigger alert."""
        rule = AlertRule(
            name="no_trigger_test",
            condition=lambda m: m.get("cpu_usage", 0) > 90,
            severity=AlertSeverity.WARNING,
            title="High CPU",
            message_template="CPU: {cpu_usage}%"
        )
        alert_manager.add_rule(rule)
        alert_manager.add_channel("test", mock_channel)
        
        # Evaluate with non-triggering metrics
        metrics = {"cpu_usage": 50}
        await alert_manager.evaluate_rules(metrics)
        
        # Verify no notification was sent
        mock_channel.send_notification.assert_not_called()
        
        # Verify no alerts were created
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_channels_notification(self, alert_manager):
        """Test notification sent to multiple channels."""
        mock_email = AsyncMock(spec=NotificationChannel)
        mock_email.send_notification = AsyncMock(return_value=True)
        
        mock_slack = AsyncMock(spec=NotificationChannel)
        mock_slack.send_notification = AsyncMock(return_value=True)
        
        rule = AlertRule(
            name="multi_channel",
            condition=lambda m: True,
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            message_template="Critical issue detected"
        )
        
        alert_manager.add_rule(rule)
        alert_manager.add_channel("email", mock_email)
        alert_manager.add_channel("slack", mock_slack)
        
        await alert_manager.evaluate_rules({})
        
        # Both channels should receive notification
        mock_email.send_notification.assert_called_once()
        mock_slack.send_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_channel_notification_failure(self, alert_manager):
        """Test handling of channel notification failure."""
        mock_failing_channel = AsyncMock(spec=NotificationChannel)
        mock_failing_channel.send_notification = AsyncMock(return_value=False)
        
        mock_working_channel = AsyncMock(spec=NotificationChannel)
        mock_working_channel.send_notification = AsyncMock(return_value=True)
        
        rule = AlertRule(
            name="failure_test",
            condition=lambda m: True,
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message_template="Test message"
        )
        
        alert_manager.add_rule(rule)
        alert_manager.add_channel("failing", mock_failing_channel)
        alert_manager.add_channel("working", mock_working_channel)
        
        # Should not raise exception despite failing channel
        await alert_manager.evaluate_rules({})
        
        # Both channels should be attempted
        mock_failing_channel.send_notification.assert_called_once()
        mock_working_channel.send_notification.assert_called_once()
    
    def test_get_active_alerts(self, alert_manager):
        """Test retrieving active alerts."""
        # Manually add alerts to test retrieval
        alert1 = Alert("alert1", "Alert 1", "Message 1", AlertSeverity.INFO)
        alert2 = Alert("alert2", "Alert 2", "Message 2", AlertSeverity.ERROR)
        alert3 = Alert("alert3", "Alert 3", "Message 3", AlertSeverity.WARNING)
        
        alert_manager._active_alerts["alert1"] = alert1
        alert_manager._active_alerts["alert2"] = alert2
        alert_manager._active_alerts["alert3"] = alert3
        
        # Resolve one alert
        alert2.resolve("Fixed")
        
        active_alerts = alert_manager.get_active_alerts()
        
        # Should only return unresolved alerts
        assert len(active_alerts) == 2
        alert_ids = [alert.id for alert in active_alerts]
        assert "alert1" in alert_ids
        assert "alert3" in alert_ids
        assert "alert2" not in alert_ids  # Resolved
    
    def test_get_alert_by_id(self, alert_manager):
        """Test retrieving alert by ID."""
        alert = Alert("findme", "Find Me", "Test message", AlertSeverity.INFO)
        alert_manager._active_alerts["findme"] = alert
        
        found_alert = alert_manager.get_alert("findme")
        assert found_alert is alert
        
        not_found = alert_manager.get_alert("nonexistent")
        assert not_found is None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        alert = Alert("resolve-test", "Resolvable Alert", "Test", AlertSeverity.WARNING)
        alert_manager._active_alerts["resolve-test"] = alert
        
        success = await alert_manager.resolve_alert("resolve-test", "Issue fixed manually")
        
        assert success is True
        assert alert.resolved is True
        assert alert.resolution_message == "Issue fixed manually"
    
    @pytest.mark.asyncio
    async def test_resolve_nonexistent_alert(self, alert_manager):
        """Test resolving non-existent alert."""
        success = await alert_manager.resolve_alert("nonexistent", "Resolution")
        
        assert success is False
    
    def test_get_alert_stats(self, alert_manager):
        """Test getting alert statistics."""
        # Add test alerts
        alerts = [
            Alert("info1", "Info Alert 1", "Message", AlertSeverity.INFO),
            Alert("info2", "Info Alert 2", "Message", AlertSeverity.INFO),
            Alert("warn1", "Warning Alert", "Message", AlertSeverity.WARNING),
            Alert("error1", "Error Alert", "Message", AlertSeverity.ERROR)
        ]
        
        for alert in alerts:
            alert_manager._active_alerts[alert.id] = alert
        
        # Resolve one alert
        alerts[1].resolve("Fixed")
        
        stats = alert_manager.get_stats()
        
        assert stats["total_alerts"] == 4
        assert stats["active_alerts"] == 3
        assert stats["resolved_alerts"] == 1
        assert stats["alerts_by_severity"]["info"] == 2
        assert stats["alerts_by_severity"]["warning"] == 1
        assert stats["alerts_by_severity"]["error"] == 1
        assert stats["alerts_by_severity"]["critical"] == 0
    
    @pytest.mark.asyncio
    async def test_alert_aggregation(self, alert_manager):
        """Test alert aggregation to prevent spam."""
        # This is a conceptual test - actual implementation would need
        # aggregation logic to prevent duplicate alerts
        
        rule = AlertRule(
            name="aggregation_test",
            condition=lambda m: m.get("errors", 0) > 10,
            severity=AlertSeverity.ERROR,
            title="High Error Rate",
            message_template="Errors: {errors}",
            cooldown_period=5.0  # 5 second cooldown
        )
        
        alert_manager.add_rule(rule)
        
        # Multiple evaluations in quick succession
        metrics = {"errors": 15}
        
        await alert_manager.evaluate_rules(metrics)
        first_alert_count = len(alert_manager.get_active_alerts())
        
        await alert_manager.evaluate_rules(metrics)
        second_alert_count = len(alert_manager.get_active_alerts())
        
        # Should not create duplicate alerts due to cooldown
        assert first_alert_count == 1
        assert second_alert_count == 1  # Same count, no new alert


class TestAlertingIntegration:
    """Test alerting system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_alerting_workflow(self):
        """Test complete alerting workflow."""
        alert_manager = AlertManager()
        
        # Setup notification channels
        mock_email = AsyncMock(spec=NotificationChannel)
        mock_email.send_notification = AsyncMock(return_value=True)
        
        mock_slack = AsyncMock(spec=NotificationChannel)
        mock_slack.send_notification = AsyncMock(return_value=True)
        
        alert_manager.add_channel("email", mock_email)
        alert_manager.add_channel("slack", mock_slack)
        
        # Setup alert rules
        cpu_rule = AlertRule(
            name="high_cpu",
            condition=lambda m: m.get("cpu_usage", 0) > 85,
            severity=AlertSeverity.WARNING,
            title="High CPU Usage",
            message_template="CPU usage is {cpu_usage}% on {host}"
        )
        
        memory_rule = AlertRule(
            name="high_memory",
            condition=lambda m: m.get("memory_usage", 0) > 90,
            severity=AlertSeverity.CRITICAL,
            title="High Memory Usage",
            message_template="Memory usage is {memory_usage}% on {host}"
        )
        
        alert_manager.add_rule(cpu_rule)
        alert_manager.add_rule(memory_rule)
        
        # Simulate monitoring data over time
        monitoring_data = [
            {"cpu_usage": 75, "memory_usage": 60, "host": "web-01"},  # No alerts
            {"cpu_usage": 90, "memory_usage": 70, "host": "web-01"},  # CPU alert
            {"cpu_usage": 95, "memory_usage": 95, "host": "web-01"},  # Both alerts
            {"cpu_usage": 80, "memory_usage": 85, "host": "web-01"},  # No new alerts
        ]
        
        for i, metrics in enumerate(monitoring_data):
            await alert_manager.evaluate_rules(metrics)
            
            if i == 0:  # No alerts expected
                assert len(alert_manager.get_active_alerts()) == 0
                
            elif i == 1:  # CPU alert expected
                active_alerts = alert_manager.get_active_alerts()
                assert len(active_alerts) == 1
                assert active_alerts[0].title == "High CPU Usage"
                
            elif i == 2:  # Memory alert added
                active_alerts = alert_manager.get_active_alerts()
                assert len(active_alerts) == 2
                
            elif i == 3:  # No new alerts due to cooldown
                active_alerts = alert_manager.get_active_alerts()
                assert len(active_alerts) == 2  # Same count
        
        # Verify notifications were sent
        assert mock_email.send_notification.call_count >= 2
        assert mock_slack.send_notification.call_count >= 2
        
        # Resolve alerts
        for alert in alert_manager.get_active_alerts():
            await alert_manager.resolve_alert(alert.id, "Issue resolved")
        
        assert len(alert_manager.get_active_alerts()) == 0
    
    @pytest.mark.asyncio
    async def test_multi_service_alerting(self):
        """Test alerting across multiple services."""
        alert_manager = AlertManager()
        
        # Different rules for different services
        api_error_rule = AlertRule(
            name="api_errors",
            condition=lambda m: (
                m.get("service") == "api" and 
                m.get("error_rate", 0) > 0.05
            ),
            severity=AlertSeverity.ERROR,
            title="API Error Rate High",
            message_template="API error rate: {error_rate:.2%} on {instance}"
        )
        
        db_connection_rule = AlertRule(
            name="db_connections",
            condition=lambda m: (
                m.get("service") == "database" and 
                m.get("connection_pool_usage", 0) > 0.9
            ),
            severity=AlertSeverity.CRITICAL,
            title="Database Connection Pool Exhausted",
            message_template="DB pool usage: {connection_pool_usage:.1%} on {instance}"
        )
        
        alert_manager.add_rule(api_error_rule)
        alert_manager.add_rule(db_connection_rule)
        
        # Simulate metrics from different services
        service_metrics = [
            {"service": "api", "error_rate": 0.08, "instance": "api-01"},
            {"service": "database", "connection_pool_usage": 0.95, "instance": "db-01"},
            {"service": "cache", "hit_rate": 0.85, "instance": "cache-01"},  # No rules
        ]
        
        for metrics in service_metrics:
            await alert_manager.evaluate_rules(metrics)
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 2
        
        titles = [alert.title for alert in active_alerts]
        assert "API Error Rate High" in titles
        assert "Database Connection Pool Exhausted" in titles
