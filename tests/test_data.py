from propensity_model_python.data import generate_customer_data


def test_generated_data_has_expected_columns() -> None:
    data = generate_customer_data(n_customers=100)

    expected_columns = {
        "customer_id",
        "age",
        "tenure_months",
        "monthly_spend",
        "previous_purchases",
        "campaign_contacts",
        "digital_customer",
        "target",
    }

    assert set(data.columns) == expected_columns


def test_generated_data_has_expected_number_of_rows() -> None:
    data = generate_customer_data(n_customers=100)

    assert len(data) == 100


def test_target_is_binary() -> None:
    data = generate_customer_data(n_customers=100)

    assert set(data["target"].unique()).issubset({0, 1})