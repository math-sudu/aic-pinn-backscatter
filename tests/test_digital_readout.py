import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from digital_readout import (adc_codes, decode_snapshot, encode_snapshot,
                             ratio_response, scan_budget, switched_bridge)
from weighted_readout import uniform_parameters


class DigitalReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
        cls.interface = json.loads((ROOT/"inputs/digital_readout.json").read_text())

    def test_six_node_kcl_with_excitation_and_reference_loading(self):
        interface = dict(self.interface, adc_input_current_scenario_a=[2e-6, -3e-6],
                         adc_reference_current_scenario_a=4e-6)
        rs, rc = 435., 501.
        result = switched_bridge(rs, rc, self.settings, interface)
        ron = interface["mux_on_resistance_ohm"]
        excitation_path = ron+interface["excitation_series_ohm"]
        reference_path = ron+interface["reference_series_ohm"]
        path = ron+interface["signal_series_filter_ohm"]
        matrix, rhs = np.zeros((6, 6)), np.zeros(6)
        # Nodes: excitation, two bridge midpoints, two ADC pins, reference.
        for a, b, resistance in [(0, 1, rc), (0, 2, rc), (1, 3, path), (2, 4, path), (0, 5, reference_path)]:
            g = 1/resistance
            matrix[a, a] += g
            matrix[b, b] += g
            matrix[a, b] -= g
            matrix[b, a] -= g
        matrix[0, 0] += 1/excitation_path
        rhs[0] = self.settings["excitation_v"]/excitation_path
        matrix[1, 1] += 1/rs
        matrix[2, 2] += 1/rc
        rhs[3:5] = -(np.asarray(interface["adc_input_current_scenario_a"])+interface["mux_bus_leakage_scenario_a"])
        rhs[5] = -interface["adc_reference_current_scenario_a"]
        nodes = np.linalg.solve(matrix, rhs)
        expected = np.r_[result["excitation"], result["bridge_nodes"], result["nodes"], result["reference"]]
        np.testing.assert_allclose(nodes, expected, rtol=1e-13)
        self.assertAlmostEqual(result["source_current"], (self.settings["excitation_v"]-nodes[0])/excitation_path, places=13)

    def test_fixed_ratio_calibration_cancels_excitation_drop_without_input_currents(self):
        interface = dict(self.interface, adc_input_current_scenario_a=[0., 0.],
                         mux_bus_leakage_scenario_a=[0., 0.], adc_reference_current_scenario_a=0.)
        raw = np.random.default_rng(68).uniform(-.001, .001, (3, 64))
        circuit = ratio_response(raw, uniform_parameters(), self.settings, interface)
        changed = ratio_response(raw, uniform_parameters(), dict(self.settings, excitation_v=2.2),
                                 dict(interface, mux_on_resistance_ohm=30.),
                                 calibration=(circuit["zero"], circuit["gain"]))
        np.testing.assert_allclose(changed["readout"], circuit["readout"], atol=1e-15)
        self.assertGreater(np.max(np.abs(changed["excitation"]-circuit["excitation"])), .3)

    def test_packet_reassembly_retains_signed_adc_words_and_rejects_mixed_states(self):
        codes = np.array([-2**23, 2**23-1, -1, 0, 1, 5, -400, 600, 777, -888, 1000, 4321])
        packets = encode_snapshot(codes, 68, 12, 100)
        self.assertTrue(all(len(packet) == 12 for packet in packets))
        identity, decoded = decode_snapshot(packets[::-1]+packets[:3])
        self.assertEqual(identity, (68, 12, 100))
        np.testing.assert_array_equal(decoded, codes)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            decode_snapshot(packets[:-1])
        with self.assertRaisesRegex(ValueError, "Mixed"):
            decode_snapshot(packets[:-1]+encode_snapshot(codes, 68, 12, 101)[-1:])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            decode_snapshot(packets+[packets[0][:-1]+b'\x01'])

    def test_adc_signed_quantization_and_range(self):
        np.testing.assert_array_equal(adc_codes(np.array([-2., -1., 0., 1., 2.])/(128*2**23), 128), [-2, -1, 0, 1, 2])
        with self.assertRaisesRegex(ValueError, "range"):
            adc_codes(1/128, 128)

    def test_scan_integrates_single_selected_bridge_and_linear_pass_charge(self):
        circuit = dict(completion=np.full(12, 500.), source_current=np.full((2, 12), .005),
                       source_power=np.full((2, 12), .0125))
        interface = dict(self.interface, bridge_excitation_capacitance_f=0., reference_capacitance_f=0.,
            signal_common_capacitance_f=0., cable_capacitance_per_signal_f=0., signal_differential_capacitance_f=0.,
            excitation_rail_capacitance_f=0., adc_mux_rail_capacitance_f=0.,
            mux_on_capacitance_f=0., mux_off_drain_capacitance_f=0.)
        # Keep a finite RC path for the separately reported settling diagnostic.
        interface["signal_common_capacitance_f"] = 1e-20
        result = scan_budget(circuit, self.settings, interface)["scans"][0]
        s = self.settings["digital_alternative"]
        duration = result["scan_ms_range"][0]/1e3
        active = (duration-interface["spi_configuration_bits"]/interface["spi_clock_hz"])/12-interface["all_banks_disabled_s"]
        q = .005*12*active+(s["analog_current_a"]+s["digital_current_a"]+
            interface["adc_external_reference_extra_current_a"]+6*interface["mux_quiescent_current_a"])*duration
        self.assertAlmostEqual(result["linear_pass_storage_mf_range"][0], q/.5*1e3, places=10)
        self.assertGreater(result["linear_pass_storage_mf_range"][0], result["ideal_conversion_storage_mf_range"][0])


if __name__ == "__main__":
    unittest.main()
