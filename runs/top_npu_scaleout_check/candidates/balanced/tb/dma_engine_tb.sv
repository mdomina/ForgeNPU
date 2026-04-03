module dma_engine_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int MAX_DIM = (ROWS > COLS) ? ROWS : COLS;

  logic clk;
  logic rst_n;
  logic dma_valid_i;
  logic dma_write_weights_i;
  logic [ADDR_WIDTH-1:0] dma_addr_i;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i;
  logic write_activations_en_o;
  logic [ADDR_WIDTH-1:0] activation_write_addr_o;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_o;
  logic write_weights_en_o;
  logic [ADDR_WIDTH-1:0] weight_write_addr_o;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_o;
  logic dma_done_o;
  logic dma_busy_o;

  dma_engine #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .dma_valid_i(dma_valid_i),
    .dma_write_weights_i(dma_write_weights_i),
    .dma_addr_i(dma_addr_i),
    .dma_payload_i(dma_payload_i),
    .write_activations_en_o(write_activations_en_o),
    .activation_write_addr_o(activation_write_addr_o),
    .activations_write_data_o(activations_write_data_o),
    .write_weights_en_o(write_weights_en_o),
    .weight_write_addr_o(weight_write_addr_o),
    .weights_write_data_o(weights_write_data_o),
    .dma_done_o(dma_done_o),
    .dma_busy_o(dma_busy_o)
  );

  always #5 clk = ~clk;

  task automatic idle_dma;
    begin
      dma_valid_i = 1'b0;
      dma_write_weights_i = 1'b0;
      dma_addr_i = '0;
      dma_payload_i = '0;
    end
  endtask

  task automatic issue_dma(
    input logic write_weights,
    input integer addr,
    input integer lane0,
    input integer lane1
  );
    begin
      idle_dma();
      dma_valid_i = 1'b1;
      dma_write_weights_i = write_weights;
      dma_addr_i = addr[ADDR_WIDTH-1:0];
      dma_payload_i[0 +: DATA_WIDTH] = lane0;
      dma_payload_i[DATA_WIDTH +: DATA_WIDTH] = lane1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic expect_idle_outputs;
    begin
      if (write_activations_en_o !== 1'b0 ||
          write_weights_en_o !== 1'b0 ||
          dma_done_o !== 1'b0) begin
        $fatal(1, "dma_engine_tb idle expectation failed");
      end
    end
  endtask

  task automatic expect_activation_write(
    input integer addr,
    input integer lane0,
    input integer lane1
  );
    begin
      if (write_activations_en_o !== 1'b1 ||
          activation_write_addr_o !== addr[ADDR_WIDTH-1:0] ||
          $signed(activations_write_data_o[0 +: DATA_WIDTH]) !== lane0 ||
          $signed(activations_write_data_o[DATA_WIDTH +: DATA_WIDTH]) !== lane1 ||
          write_weights_en_o !== 1'b0 ||
          dma_done_o !== 1'b1) begin
        $fatal(1, "dma_engine_tb activation write expectation failed");
      end
    end
  endtask

  task automatic expect_weight_write(
    input integer addr,
    input integer lane0,
    input integer lane1
  );
    begin
      if (write_weights_en_o !== 1'b1 ||
          weight_write_addr_o !== addr[ADDR_WIDTH-1:0] ||
          $signed(weights_write_data_o[0 +: DATA_WIDTH]) !== lane0 ||
          $signed(weights_write_data_o[DATA_WIDTH +: DATA_WIDTH]) !== lane1 ||
          write_activations_en_o !== 1'b0 ||
          dma_done_o !== 1'b1) begin
        $fatal(1, "dma_engine_tb weight write expectation failed");
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    idle_dma();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    issue_dma(1'b0, 1, 3, 4);
    expect_activation_write(1, 3, 4);
    idle_dma();
    @(posedge clk);
    #1;
    expect_idle_outputs();

    issue_dma(1'b1, 2, 7, 8);
    expect_weight_write(2, 7, 8);
    idle_dma();
    @(posedge clk);
    #1;
    expect_idle_outputs();

    $display("dma_engine_tb passed");
    $finish;
  end
endmodule
